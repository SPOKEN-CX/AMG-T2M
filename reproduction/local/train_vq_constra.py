import argparse
import os
import json
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

warnings.filterwarnings('ignore')
torch.manual_seed(42)

import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer

from models.Retrival.text_motion_sim import *

warnings.simplefilter("ignore", FutureWarning)


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


def weighted_info_nce_loss(anchor, positives, negatives, pos_weights, temperature):
    """
    加权的 InfoNCE 损失函数（单样本版本）。
    【保留原逻辑做参考，不在批量化中调用此函数】
    参数:
    - anchor: [C] 张量，锚点特征向量。
    - positives: [x, C] 张量，多个正样本的特征向量。
    - negatives: [x, C] 张量，多个负样本的特征向量。
    - pos_weights: [x] 张量，每个正样本的权重。
    - temperature: 标量，温度参数。

    返回:
    - loss: 标量，计算得到的加权 InfoNCE 损失。
    """
    # 归一化处理以使用余弦相似度
    anchor_norm = torch.nn.functional.normalize(anchor, p=2, dim=0)  # [C]
    positives_norm = torch.nn.functional.normalize(positives, p=2, dim=1)  # [x, C]
    negatives_norm = torch.nn.functional.normalize(negatives, p=2, dim=1)  # [x, C]

    # 计算正样本与锚点的相似度 [x]
    pos_sim = torch.matmul(positives_norm, anchor_norm)  # [x]
    # 计算负样本与锚点的相似度 [x]
    neg_sim = torch.matmul(negatives_norm, anchor_norm)  # [x]

    # 计算权重，使用传入的 pos_weights
    weights = pos_weights  # [x]

    # 计算分子和分母
    numerator = torch.sum(weights * torch.exp(pos_sim / temperature))  # 标量
    denominator = numerator + torch.sum(torch.exp(neg_sim / temperature))  # 标量

    # 计算损失
    loss = -torch.log(numerator / denominator)

    return loss


def weighted_info_nce_loss_batched(anchors, pos_samples, neg_samples, pos_weights, temperature):
    """
    批量(Batch)方式计算加权 InfoNCE 损失。

    参数:
    - anchors: [B, C]，一批样本各自的 anchor 向量
    - pos_samples: [B, x, C]，每个样本对应 x 个正样本
    - neg_samples: [B, x, C]，每个样本对应 x 个负样本
    - pos_weights: [B, x]，每个样本对应 x 个正样本的权重
    - temperature: float，温度系数

    返回:
    - loss: 标量 (已经在 B 维上做平均)
    """
    # 1) 归一化
    anchors_norm = torch.nn.functional.normalize(anchors, p=2, dim=1)      # [B, C]
    pos_norm = torch.nn.functional.normalize(pos_samples, p=2, dim=2)     # [B, x, C]
    neg_norm = torch.nn.functional.normalize(neg_samples, p=2, dim=2)     # [B, x, C]

    # 2) 计算余弦相似度
    #    pos_sim[i, j] = (anchors_norm[i] * pos_norm[i, j]).sum(dim=-1)
    pos_sim = torch.sum(anchors_norm.unsqueeze(1) * pos_norm, dim=2)      # [B, x]
    neg_sim = torch.sum(anchors_norm.unsqueeze(1) * neg_norm, dim=2)      # [B, x]

    # 3) 计算分子和分母
    numerator = torch.sum(pos_weights * torch.exp(pos_sim / temperature), dim=1)  # [B]
    denominator = numerator + torch.sum(torch.exp(neg_sim / temperature), dim=1)  # [B]

    # 4) InfoNCE 损失
    loss = -torch.log(numerator / denominator)  # [B]
    return loss.mean()                          # 标量 (对 B 取均值)


if __name__ == '__main__':
    ##### ---- Exp dirs ---- #####
    args = option_vq.get_args_parser()
    args.check_dir ='./checkpoints_model/'
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

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

    logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    train_loader = dataset_VQ.DATALoader(args.dataname,
                                         args.batch_size,
                                         window_size=args.window_size,
                                         unit_length=2 ** args.down_t)

    train_loader_iter = dataset_VQ.cycle(train_loader)
    if args.dataname == 'BABEL_TEACH':
        opt = argparse.Namespace()
        opt.max_motion_length = 196
        opt.joints_num = 22
        opt.data_root = '../Dataset/BABEL_TEACH'
        opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.mul_data_size = 2
        opt.unit_length = 2 ** args.down_t
        opt.max_text_len = 20
        opt.meta_dir = './checkpoints/BABEL_TEACH/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test_2000.txt')
        test_dataset = dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file,
                                                                                w_vectorizer, )
        val_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True, shuffle=True,
                                collate_fn=dataset_TM_eval.collate_fn)
        print('text_dataloader successfully loaded!')
    else:
        val_loader = dataset_TM_eval.DATALoader(args.dataname, True,
                                                32,
                                                w_vectorizer,
                                                unit_length=2 ** args.down_t)
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
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    start_iter = 0  # 记录从第几个 iter 开始训练

    if args.resume_pth:
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cuda')

        net.load_state_dict(ckpt['net'], strict=True)
        logger.info("Network parameters loaded.")

        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info("Optimizer state loaded from checkpoint.")
            for i, group in enumerate(optimizer.param_groups):
                for j, p in enumerate(group['params']):
                    if p in optimizer.state:
                        state_param = optimizer.state[p]
                        for key, value in state_param.items():
                            if torch.is_tensor(value):
                                state_param[key] = value.to('cuda')

        # 加载 scheduler
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info("Scheduler state loaded from checkpoint.")

        # 如果 checkpoint 中保存了 iter 信息，就从对应的 iter 开始继续
        start_iter = ckpt.get('iter', 0)
        logger.info(f"Resumed from iter {start_iter}")

    net.train()
    net.cuda()

    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    # 对比学习参数
    args.x_pos_neg = 150  # 每个样本选多少正/负样本
    args.contrastive_weight = 0.005  # 对比损失权重
    args.contrastive_temp = 0.07  # InfoNCE温度参数

    ##### ------ Warm-up 仅当从头开始训练时进行 ------ #####
    #如果 args.resume_pth 为空，说明是从头开始训练，那么就进行 warm-up
    if not args.resume_pth:
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        for nb_iter in range(1, args.warm_up_iter + 1):
            optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

            gt_motion, text, motion_retrival, text_retrival = next(train_loader_iter)
            gt_motion = gt_motion.cuda().float()  # (bs, seq_len, dim)

            pred_motion, loss_commit, perplexity, quantized = net(gt_motion)

            # 计算重建和速度损失
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

                logger.info(
                    f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} "
                    f"\t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
                )

                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net, logger, writer, start_iter,
            best_fid=1000, best_iter=0, best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper
        )

    for nb_iter in range(start_iter + 1, args.total_iter + 1):

        total_start = time.time()
        gt_motion, text, motion_retrival, text_retrival = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()

        pred_motion, loss_commit, perplexity, quantized = net(gt_motion)  # quantized: [B, C]

        # 1) 计算 VQ-VAE 的损失 (重建 + 速度 + commit)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = 4 * loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        # 2) 对比学习部分（批量处理）
        text_retrival = text_retrival.squeeze(1)
        motion_retrival = motion_retrival.squeeze(1)

        # 在 GPU 上获得相似度矩阵
        score_matrix = text_motion_sim_V3(text_retrival, motion_retrival)  # [B, B], GPU tensor
        B = score_matrix.size(0)

        # 对角线设为 -inf
        indices = torch.arange(B, device=score_matrix.device)
        score_matrix[indices, indices] = float('-inf')

        x = args.x_pos_neg
        top_values, top_indices = torch.topk(score_matrix, k=x, dim=1)       # [B, x], [B, x]
        bottom_values, bottom_indices = torch.topk(score_matrix, k=x, dim=1, largest=False)  # [B, x], [B, x]

        # anchors = quantized: [B, C]
        anchors = quantized
        con_start=time.time()
        # pos_samples, neg_samples => [B, x, C]
        pos_samples = quantized[top_indices]
        neg_samples = quantized[bottom_indices]

        # pos_scores = top_values => [B, x]
        pos_scores = torch.tensor(top_values).to("cuda")

        # softmax 得到权重 => [B, x]
        pos_weights = torch.nn.functional.softmax(pos_scores / args.contrastive_temp, dim=1)
        #pos_weights = pos_weights.to(pos_scores.device)
        con_end = time.time()
        # 一次性计算 batched InfoNCE
        contrastive_loss = weighted_info_nce_loss_batched(
            anchors, pos_samples, neg_samples, pos_weights, args.contrastive_temp
        )

        # 与主体损失合并
        loss += args.contrastive_weight * contrastive_loss

        # 3) 反向传播 & 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 4) 累计日志并打印
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        total_end = time.time()
        # print(f'nb_iter:{nb_iter}'
        #         f'total time: {total_end - total_start}'
        #       f'constra time: {con_end - con_start}')


        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            writer.add_scalar('./Train/ContrastiveLoss', contrastive_loss.item(), nb_iter)

            logger.info(
                f"Train. Iter {nb_iter} : Commit. {avg_commit:.5f}, PPL. {avg_perplexity:.2f}, "
                f"Recons. {avg_recons:.5f}, Contrastive: {contrastive_loss.item():.5f}"
            )

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

        # 5) 保存 & 评估
        if nb_iter % args.save_iter == 0:
            checkpoint_path = os.path.join(args.check_dir, args.exp_name, f"model_{nb_iter}.pth")
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iter': nb_iter
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at {checkpoint_path}")

        if nb_iter % args.eval_iter == 0:
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
                eval_trans.evaluation_vqvae(
                    args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter,
                    best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper
                )
