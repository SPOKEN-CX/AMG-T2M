import argparse
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from os.path import join as pjoin
import time

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr

def main():
    ##### ---- 初始化进程组 ---- #####
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    ##### ---- 解析参数 ---- #####
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)

    # 仅在主进程创建输出目录和日志记录器
    if dist.get_rank() == 0:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
        os.makedirs(args.out_dir, exist_ok = True)
        logger = utils_model.get_logger(args.out_dir)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    else:
        logger = None
        writer = None

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    if args.dataname == 'kit' :
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 21
    elif args.dataname =='BABEL_TEACH':
        dataset_opt_path = './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt'
        args.nb_joints = 22
    else :
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    if logger is not None:
        logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    # 创建数据集
    train_dataset = dataset_VQ.VQMotionDataset(args.dataname,
                                                  window_size=args.window_size,
                                                  unit_length=2**args.down_t)

    # 创建 DistributedSampler
    train_sampler = DistributedSampler(train_dataset)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=4,
                              drop_last=False,
                              pin_memory=True)

    train_loader_iter = dataset_VQ.cycle(train_loader)

    if args.dataname == 'BABEL_TEACH':
        opt = argparse.Namespace()
        opt.max_motion_length=196
        opt.joints_num=22
        opt.data_root='../Dataset/BABEL_TEACH'
        opt.motion_dir= pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir=pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.mul_data_size = 2
        opt.unit_length = 2**args.down_t
        opt.max_text_len = 20
        opt.meta_dir='./checkpoints/BABEL_TEACH/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test_2000.txt')
        test_dataset=dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file, w_vectorizer)
        val_sampler = DistributedSampler(test_dataset, shuffle=False)
        val_loader=DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True, sampler=val_sampler, collate_fn=dataset_TM_eval.collate_fn)
        if logger is not None:
            logger.info('Validation dataloader successfully loaded!')
    else:
        val_loader = None  # 请根据您的实际情况进行修改

    ##### ---- Network ---- #####
    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                           args.nb_code,
                           args.code_dim,
                           args.output_emb_width,
                           args.down_t,
                           args.stride_t,
                           args.width,
                           args.depth,
                           args.dilation_growth_rate,
                           args.vq_act,
                           args.vq_norm)

    if args.resume_pth :
        if logger is not None:
            logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.cuda(local_rank)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    net.train()

    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    ##### ------ warm-up ------- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    for nb_iter in range(1, args.warm_up_iter + 1):
        train_sampler.set_epoch(nb_iter)
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda(local_rank).float()
        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            avg_recons_tensor = torch.tensor(avg_recons).cuda(local_rank)
            avg_perplexity_tensor = torch.tensor(avg_perplexity).cuda(local_rank)
            avg_commit_tensor = torch.tensor(avg_commit).cuda(local_rank)

            dist.all_reduce(avg_recons_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_perplexity_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_commit_tensor, op=dist.ReduceOp.SUM)

            avg_recons = avg_recons_tensor.item() / dist.get_world_size()
            avg_perplexity = avg_perplexity_tensor.item() / dist.get_world_size()
            avg_commit = avg_commit_tensor.item() / dist.get_world_size()

            if dist.get_rank() == 0:
                if writer is not None:
                    writer.add_scalar('Warmup/L1', avg_recons, nb_iter)
                    writer.add_scalar('Warmup/PPL', avg_perplexity, nb_iter)
                    writer.add_scalar('Warmup/Commit', avg_commit, nb_iter)
                logger.info(
                    f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    print('2222')
    if dist.get_rank() == 0:
        print('开始进行evaluation！')
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net.module, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100,
            best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
        print('evaluation_vqvae success!!!!!!!')
    else:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100

    for nb_iter in range(1, args.total_iter + 1):

        train_sampler.set_epoch(nb_iter)

        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda(local_rank).float() # bs, nb_joints, joints_dim, seq_len

        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)

        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            # 汇总各个进程的平均损失
            avg_recons_tensor = torch.tensor(avg_recons).cuda(local_rank)
            avg_perplexity_tensor = torch.tensor(avg_perplexity).cuda(local_rank)
            avg_commit_tensor = torch.tensor(avg_commit).cuda(local_rank)

            dist.all_reduce(avg_recons_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_perplexity_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_commit_tensor, op=dist.ReduceOp.SUM)

            avg_recons = avg_recons_tensor.item() / dist.get_world_size()
            avg_perplexity = avg_perplexity_tensor.item() / dist.get_world_size()
            avg_commit = avg_commit_tensor.item() / dist.get_world_size()

            if dist.get_rank() == 0:
                if writer is not None:
                    writer.add_scalar('Train/L1', avg_recons, nb_iter)
                    writer.add_scalar('Train/PPL', avg_perplexity, nb_iter)
                    writer.add_scalar('Train/Commit', avg_commit, nb_iter)
                logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

        if nb_iter % args.eval_iter==0 and dist.get_rank() == 0:
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
                args.out_dir, val_loader, net.module, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1,
                best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)

    # 训练结束，销毁进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

# from torch.utils.data import DataLoader, DistributedSampler
#
# def main():
#     ##### ---- 初始化进程组 ---- #####
#     dist.init_process_group(backend='nccl')
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#
#     ##### ---- 解析参数 ---- #####
#     args = option_vq.get_args_parser()
#     torch.manual_seed(args.seed)
#
#     # 仅在主进程创建输出目录和日志记录器
#     if dist.get_rank() == 0:
#         args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
#         os.makedirs(args.out_dir, exist_ok=True)
#         logger = utils_model.get_logger(args.out_dir)
#         writer = SummaryWriter(args.out_dir)
#         logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
#     else:
#         logger = None
#         writer = None
#
#     w_vectorizer = WordVectorizer('./glove', 'our_vab')
#
#     if args.dataname == 'kit':
#         dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
#         args.nb_joints = 21
#     elif args.dataname == 'BABEL_TEACH':
#         dataset_opt_path = './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt'
#         args.nb_joints = 22
#     else:
#         dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
#         args.nb_joints = 22
#
#     if logger is not None:
#         logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')
#
#     wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
#     eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
#
#     ##### ---- Dataloader ---- #####
#     # 创建数据集
#     train_dataset = dataset_VQ.VQMotionDataset(args.dataname,
#                                                window_size=args.window_size,
#                                                unit_length=2**args.down_t)
#
#     # 创建 DistributedSampler
#     train_sampler = DistributedSampler(train_dataset, shuffle=True)
#
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset,
#                               batch_size=args.batch_size,
#                               sampler=train_sampler,
#                               num_workers=4,  # 已调整为4
#                               drop_last=False,
#                               pin_memory=True)
#
#     # 不使用 cycle 函数，按 epoch 迭代
#     train_loader_iter = iter(train_loader)
#
#     # 验证数据加载器
#     if args.dataname == 'BABEL_TEACH':
#         opt = argparse.Namespace()
#         opt.max_motion_length = 196
#         opt.joints_num = 22
#         opt.data_root = '../Dataset/BABEL_TEACH'
#         opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
#         opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
#         opt.mul_data_size = 2
#         opt.unit_length = 2**args.down_t
#         opt.max_text_len = 20
#         opt.meta_dir = './checkpoints/BABEL_TEACH/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
#         mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
#         std = np.load(pjoin(opt.meta_dir, 'std.npy'))
#         w_vectorizer = WordVectorizer('./glove', 'our_vab')
#         split_file = pjoin(opt.data_root, 'test_2000.txt')
#         test_dataset = dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file, w_vectorizer)
#         val_sampler = DistributedSampler(test_dataset, shuffle=False)
#         val_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True, sampler=val_sampler, collate_fn=dataset_TM_eval.collate_fn)
#         if logger is not None:
#             logger.info('Validation dataloader successfully loaded!')
#     else:
#         val_loader = None  # 请根据您的实际情况进行修改
#
#     ##### ---- Network ---- #####
#     net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
#                            args.nb_code,
#                            args.code_dim,
#                            args.output_emb_width,
#                            args.down_t,
#                            args.stride_t,
#                            args.width,
#                            args.depth,
#                            args.dilation_growth_rate,
#                            args.vq_act,
#                            args.vq_norm)
#
#     if args.resume_pth:
#         if logger is not None:
#             logger.info('loading checkpoint from {}'.format(args.resume_pth))
#         ckpt = torch.load(args.resume_pth, map_location='cpu')
#         net.load_state_dict(ckpt['net'], strict=True)
#     net.cuda(local_rank)
#     net = DDP(net, device_ids=[local_rank], output_device=local_rank)
#     net.train()
#
#     ##### ---- Optimizer & Scheduler ---- #####
#     optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
#
#     Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)
#
#     ##### ------ warm-up ------- #####
#     avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
#
#     for nb_iter in range(1, args.warm_up_iter + 1):
#         start_iter = time.time()
#
#         # 设置 epoch
#         iter_start = time.time()
#         train_sampler.set_epoch(nb_iter)
#         iter_end = time.time()
#         print(f"Iter {nb_iter}: set_epoch time: {iter_end - iter_start:.4f}s")
#
#         # 学习率更新
#         lr_start = time.time()
#         optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
#         lr_end = time.time()
#         print(f"Iter {nb_iter}: update_lr_warm_up time: {lr_end - lr_start:.4f}s")
#
#         # 数据加载
#         data_start = time.time()
#         try:
#             gt_motion = next(train_loader_iter)
#         except StopIteration:
#             train_loader_iter = iter(train_loader)
#             gt_motion = next(train_loader_iter)
#         data_end = time.time()
#         print(f"Iter {nb_iter}: data loading time: {data_end - data_start:.4f}s")
#
#         # 数据传输到GPU
#         transfer_start = time.time()
#         gt_motion = gt_motion.cuda(local_rank).float()
#         transfer_end = time.time()
#         print(f"Iter {nb_iter}: data transfer time: {transfer_end - transfer_start:.4f}s")
#
#         # 前向传播
#         forward_start = time.time()
#         pred_motion, loss_commit, perplexity = net(gt_motion)
#         forward_end = time.time()
#         print(f"Iter {nb_iter}: forward pass time: {forward_end - forward_start:.4f}s")
#
#         # 计算损失
#         loss_start = time.time()
#         loss_motion = Loss(pred_motion, gt_motion)
#         loss_vel = Loss.forward_vel(pred_motion, gt_motion)
#         loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
#         loss_end = time.time()
#         print(f"Iter {nb_iter}: loss computation time: {loss_end - loss_start:.4f}s")
#
#         # 反向传播和优化
#         backward_start = time.time()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         backward_end = time.time()
#         print(f"Iter {nb_iter}: backward pass and optimizer step time: {backward_end - backward_start:.4f}s")
#
#         # 累积损失
#         avg_recons += loss_motion.item()
#         avg_perplexity += perplexity.item()
#         avg_commit += loss_commit.item()
#
#         # 打印日志
#         if nb_iter % args.print_iter == 0:
#             print_iter_start = time.time()
#             avg_recons /= args.print_iter
#             avg_perplexity /= args.print_iter
#             avg_commit /= args.print_iter
#
#             # 汇总各个进程的平均损失
#             avg_recons_tensor = torch.tensor(avg_recons).cuda(local_rank)
#             avg_perplexity_tensor = torch.tensor(avg_perplexity).cuda(local_rank)
#             avg_commit_tensor = torch.tensor(avg_commit).cuda(local_rank)
#
#             dist.all_reduce(avg_recons_tensor, op=dist.ReduceOp.SUM)
#             dist.all_reduce(avg_perplexity_tensor, op=dist.ReduceOp.SUM)
#             dist.all_reduce(avg_commit_tensor, op=dist.ReduceOp.SUM)
#
#             avg_recons = avg_recons_tensor.item() / dist.get_world_size()
#             avg_perplexity = avg_perplexity_tensor.item() / dist.get_world_size()
#             avg_commit = avg_commit_tensor.item() / dist.get_world_size()
#
#             if dist.get_rank() == 0:
#                 log_start = time.time()
#                 if writer is not None:
#                     writer.add_scalar('Warmup/L1', avg_recons, nb_iter)
#                     writer.add_scalar('Warmup/PPL', avg_perplexity, nb_iter)
#                     writer.add_scalar('Warmup/Commit', avg_commit, nb_iter)
#                 logger.info(
#                     f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
#                 log_end = time.time()
#                 print(f"Iter {nb_iter}: logging time: {log_end - log_start:.4f}s")
#
#             avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
#             print_iter_end = time.time()
#             print(f"Iter {nb_iter}: print_iter time: {print_iter_end - print_iter_start:.4f}s")
#
#         end_iter = time.time()
#         print(f"Iter {nb_iter}: total iteration time: {end_iter - start_iter:.4f}s")
#
#     ##### ---- Training ---- #####
#     avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
#     print('开始进行evaluation！')
#     if dist.get_rank() == 0:
#         print('开始进行evaluation！')
#         best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
#             args.out_dir, val_loader, net.module, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100,
#             best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
#         print('evaluation_vqvae success!!!!!!!')
#     else:
#         best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100
#
#     for nb_iter in range(1, args.total_iter + 1):
#
#         train_sampler.set_epoch(nb_iter)
#
#         try:
#             gt_motion = next(train_loader_iter)
#         except StopIteration:
#             train_loader_iter = iter(train_loader)
#             gt_motion = next(train_loader_iter)
#         gt_motion = gt_motion.cuda(local_rank).float() # bs, nb_joints, joints_dim, seq_len
#
#         pred_motion, loss_commit, perplexity = net(gt_motion)
#         loss_motion = Loss(pred_motion, gt_motion)
#         loss_vel = Loss.forward_vel(pred_motion, gt_motion)
#
#         loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#         avg_recons += loss_motion.item()
#         avg_perplexity += perplexity.item()
#         avg_commit += loss_commit.item()
#
#         if nb_iter % args.print_iter ==  0 :
#             avg_recons /= args.print_iter
#             avg_perplexity /= args.print_iter
#             avg_commit /= args.print_iter
#
#             # 汇总各个进程的平均损失
#             avg_recons_tensor = torch.tensor(avg_recons).cuda(local_rank)
#             avg_perplexity_tensor = torch.tensor(avg_perplexity).cuda(local_rank)
#             avg_commit_tensor = torch.tensor(avg_commit).cuda(local_rank)
#
#             dist.all_reduce(avg_recons_tensor, op=dist.ReduceOp.SUM)
#             dist.all_reduce(avg_perplexity_tensor, op=dist.ReduceOp.SUM)
#             dist.all_reduce(avg_commit_tensor, op=dist.ReduceOp.SUM)
#
#             avg_recons = avg_recons_tensor.item() / dist.get_world_size()
#             avg_perplexity = avg_perplexity_tensor.item() / dist.get_world_size()
#             avg_commit = avg_commit_tensor.item() / dist.get_world_size()
#
#             if dist.get_rank() == 0:
#                 if writer is not None:
#                     writer.add_scalar('Train/L1', avg_recons, nb_iter)
#                     writer.add_scalar('Train/PPL', avg_perplexity, nb_iter)
#                     writer.add_scalar('Train/Commit', avg_commit, nb_iter)
#                 logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
#
#             avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
#
#         if nb_iter % args.eval_iter == 0 and dist.get_rank() == 0:
#             best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
#                 args.out_dir, val_loader, net.module, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1,
#                 best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
#
#     # 训练结束，销毁进程组
#     dist.destroy_process_group()
#
# if __name__ == '__main__':
#     main()