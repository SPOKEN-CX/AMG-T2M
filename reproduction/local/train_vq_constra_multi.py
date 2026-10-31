import argparse
import os
import json
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore')
torch.manual_seed(42)  # 如果需要固定随机种子

import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from models.Retrival.text_motion_sim import text_motion_sim

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr

def info_nce_loss(anchor, positives, negatives, temperature):
    """
    使用 InfoNCE loss:
    anchor: [C]
    positives: [x, C]
    negatives: [x, C]

    将多个正样本平均成一个正样本中心 pos_centroid: [C]
    """
    anchor_norm = anchor / anchor.norm(dim=0, keepdim=True)
    positives_norm = positives / positives.norm(dim=1, keepdim=True)
    negatives_norm = negatives / negatives.norm(dim=1, keepdim=True)

    # 正样本中心
    pos_centroid = positives_norm.mean(dim=0)  # [C]
    pos_sim = torch.sum(anchor_norm * pos_centroid, dim=0, keepdim=True)  # [1]

    # 负样本相似度
    neg_sim = (anchor_norm.unsqueeze(0) * negatives_norm).sum(dim=1)  # [x]

    # 拼接logits
    logits = torch.cat([pos_sim, neg_sim], dim=0)  # [1 + x]
    labels = torch.zeros(1, dtype=torch.long, device=logits.device)  # 第一个为正样本

    # 温度缩放
    logits = logits / temperature
    # 调整维度以匹配 CrossEntropyLoss 接口
    logits = logits.unsqueeze(0)  # [1, 1+x]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    return loss

def main():
    ##### ---- 初始化进程组 ---- #####
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    ##### ---- 解析参数 ---- #####
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)

    # 仅在主进程创建输出目录和日志记录器
    if dist.get_rank() == 0:
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
        os.makedirs(args.out_dir, exist_ok=True)
        logger = utils_model.get_logger(args.out_dir)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    else:
        logger = None
        writer = None

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    if args.dataname == 'kit':
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 21
    elif args.dataname == 'BABEL_TEACH':
        dataset_opt_path = './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt'
        args.nb_joints = 22
    else:
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    if logger is not None:
        logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    # 创建训练集和分布式采样器
    train_dataset = dataset_VQ.VQMotionDataset_V1(args.dataname,
                                               window_size=args.window_size,
                                               unit_length=2**args.down_t)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=4,
                              drop_last=False,
                              pin_memory=True)
    train_loader_iter = dataset_VQ.cycle(train_loader)

    # 验证集
    if args.dataname == 'BABEL_TEACH':
        opt = argparse.Namespace()
        opt.max_motion_length = 196
        opt.joints_num = 22
        opt.data_root = '../Dataset/BABEL_TEACH'
        opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.mul_data_size = 2
        opt.unit_length = 2**args.down_t
        opt.max_text_len = 20
        opt.meta_dir = './checkpoints/BABEL_TEACH/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test_2000.txt')
        test_dataset = dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file,
                                                                                w_vectorizer)
        val_sampler = DistributedSampler(test_dataset, shuffle=False)
        val_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True, sampler=val_sampler,
                                collate_fn=dataset_TM_eval.collate_fn)
        if logger is not None:
            logger.info('Validation dataloader successfully loaded!')
    else:
        val_loader = None

    ##### ---- Network ---- #####
    net = vqvae.HumanVQVAE(args,
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

    if args.resume_pth and logger is not None:
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)

    net.cuda(local_rank)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)
    net.train()

    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    # 对比学习参数
    args.x_pos_neg = 50  # 每个样本选多少正/负样本
    args.contrastive_weight = 0.01  # 对比损失权重
    args.contrastive_temp = 0.07  # InfoNCE温度参数

    ##### ------ warm-up ------- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    for nb_iter in range(1, args.warm_up_iter + 1):
        train_sampler.set_epoch(nb_iter)
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

        gt_motion, text = next(train_loader_iter)
        gt_motion = gt_motion.cuda(local_rank).float()

        pred_motion, loss_commit, perplexity, projected_features = net(gt_motion)
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
            # 聚合多卡的结果
            avg_recons_tensor = torch.tensor(avg_recons / args.print_iter).cuda(local_rank)
            avg_perplexity_tensor = torch.tensor(avg_perplexity / args.print_iter).cuda(local_rank)
            avg_commit_tensor = torch.tensor(avg_commit / args.print_iter).cuda(local_rank)

            dist.all_reduce(avg_recons_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_perplexity_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_commit_tensor, op=dist.ReduceOp.SUM)

            avg_recons_all = avg_recons_tensor.item() / dist.get_world_size()
            avg_perplexity_all = avg_perplexity_tensor.item() / dist.get_world_size()
            avg_commit_all = avg_commit_tensor.item() / dist.get_world_size()

            if dist.get_rank() == 0:
                if writer is not None:
                    writer.add_scalar('Warmup/L1', avg_recons_all, nb_iter)
                    writer.add_scalar('Warmup/PPL', avg_perplexity_all, nb_iter)
                    writer.add_scalar('Warmup/Commit', avg_commit_all, nb_iter)
                logger.info(
                    f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit_all:.5f} \t PPL. {avg_perplexity_all:.2f} \t Recons.  {avg_recons_all:.5f}")

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Evaluation (before training) ---- #####
    # 仅在主进程上进行evaluation
    if dist.get_rank() == 0 and val_loader is not None:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net.module, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100,
            best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
    else:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    for nb_iter in range(1, args.total_iter + 1):
        train_sampler.set_epoch(nb_iter)

        gt_motion, text = next(train_loader_iter)
        gt_motion = gt_motion.cuda(local_rank).float()

        pred_motion, loss_commit, perplexity, projected_features = net(gt_motion)

        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        # 对比学习
        motions_list = [m.cpu().numpy() for m in gt_motion]
        score_matrix = text_motion_sim(text, motions_list, device='cuda')
        np.fill_diagonal(score_matrix, -np.inf)
        B = score_matrix.shape[0]
        x = args.x_pos_neg
        score_matrix_t = torch.from_numpy(score_matrix).float().cuda(local_rank)
        top_values, top_indices = torch.topk(score_matrix_t, k=x, dim=1)
        bottom_values, bottom_indices = torch.topk(score_matrix_t, k=x, dim=1, largest=False)

        contrastive_loss_total = 0.0
        for i in range(B):
            anchor = projected_features[i]
            pos_samples = projected_features[top_indices[i]]
            neg_samples = projected_features[bottom_indices[i]]

            loss_i = info_nce_loss(anchor, pos_samples, neg_samples, args.contrastive_temp)
            contrastive_loss_total += loss_i
        contrastive_loss_total = contrastive_loss_total / B
        contrastive_loss_total /= args.x_pos_neg
        loss += args.contrastive_weight * contrastive_loss_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter == 0:
            # 汇总损失
            avg_recons_tensor = torch.tensor(avg_recons / args.print_iter).cuda(local_rank)
            avg_perplexity_tensor = torch.tensor(avg_perplexity / args.print_iter).cuda(local_rank)
            avg_commit_tensor = torch.tensor(avg_commit / args.print_iter).cuda(local_rank)
            contrastive_loss_tensor = torch.tensor(contrastive_loss_total.item()).cuda(local_rank)

            dist.all_reduce(avg_recons_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_perplexity_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_commit_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(contrastive_loss_tensor, op=dist.ReduceOp.SUM)

            avg_recons_all = avg_recons_tensor.item() / dist.get_world_size()
            avg_perplexity_all = avg_perplexity_tensor.item() / dist.get_world_size()
            avg_commit_all = avg_commit_tensor.item() / dist.get_world_size()
            contrastive_loss_all = contrastive_loss_tensor.item() / dist.get_world_size()

            if dist.get_rank() == 0:
                if writer is not None:
                    writer.add_scalar('./Train/L1', avg_recons_all, nb_iter)
                    writer.add_scalar('./Train/PPL', avg_perplexity_all, nb_iter)
                    writer.add_scalar('./Train/Commit', avg_commit_all, nb_iter)
                    writer.add_scalar('./Train/ContrastiveLoss', contrastive_loss_all, nb_iter)

                logger.info(
                    f"Train. Iter {nb_iter} : Commit. {avg_commit_all:.5f}, PPL. {avg_perplexity_all:.2f}, Recons. {avg_recons_all:.5f}, Contrastive: {contrastive_loss_all:.5f}")

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

        if nb_iter % args.eval_iter == 0 and dist.get_rank() == 0 and val_loader is not None:
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
                args.out_dir, val_loader, net.module, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1,
                best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper
            )

    # 训练结束，销毁进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
