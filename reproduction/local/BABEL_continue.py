import argparse
import os
import json
import warnings
import numpy as np
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

from models.Retrival.text_motion_sim import text_motion_sim,text_motion_sim_V1
from models.Retrival.src.config import read_config
from models.Retrival.src.load import load_model_from_cfg
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
warnings.simplefilter("ignore", FutureWarning)

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

    我们将多个正样本平均成一个正样本中心 pos_centroid: [C]
    然后计算 sim(anchor, pos_centroid) 和 sim(anchor, negatives[i])。
    positives 数量为 x，negatives 数量为 x。

    logits = [sim(a, p), sim(a, n1), sim(a, n2), ... sim(a, nx)]
    label为0表示第一个是positive。
    """
    # 归一化处理以用cosine相似度（可选）
    anchor_norm = anchor / anchor.norm(dim=0, keepdim=True)
    positives_norm = positives / positives.norm(dim=1, keepdim=True)
    negatives_norm = negatives / negatives.norm(dim=1, keepdim=True)

    # 计算正样本中心
    pos_centroid = positives_norm.mean(dim=0)  # [C]
    pos_sim = torch.sum(anchor_norm * pos_centroid, dim=0, keepdim=True)  # [1]

    # 计算负样本相似度
    neg_sim = (anchor_norm.unsqueeze(0) * negatives_norm).sum(dim=1)  # [x]

    # 拼接logits: 正样本在前，负样本在后
    logits = torch.cat([pos_sim, neg_sim], dim=0)  # [1 + x]

    # InfoNCE的标签，第一个为正样本
    labels = torch.zeros(1, dtype=torch.long, device=logits.device)

    # 温度缩放
    logits = logits / temperature

    # CrossEntropyLoss要求 logits: [N, C], labels: [N]，这里N=1时，需要扩展batch维度
    # 我们这里一共 1+x 个候选，其中 pos 在 index=0
    logits = logits.unsqueeze(0)  # [1, 1+x]
    labels = labels  # [1]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)  # 计算单个样本的 InfoNCE loss
    return loss


if __name__ == '__main__':
    ##### ---- Exp dirs ---- #####
    args = option_vq.get_args_parser()
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
        test_dataset = dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(
            opt, mean, std, split_file, w_vectorizer,
        )
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

    # 只加载网络参数，其余从头开始
    if args.resume_pth:
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cuda:0')
        net.load_state_dict(ckpt['net'], strict=True)
        logger.info('Network weights loaded from checkpoint (optimizer/scheduler not loaded).')

    net.train()
    net.cuda()

    ##### ---- Optimizer & Scheduler ---- #####
    # 从头初始化 optimizer/scheduler
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    # 对比学习参数
    args.x_pos_neg = 50  # 每个样本选多少正/负样本
    args.contrastive_weight = 0.05  # 对比损失权重
    args.contrastive_temp = 0.07  # InfoNCE温度参数

    ##### ---- load_constra_model ---- #####
    run_dir = './models/Retrival/models/tmr_humanml3d_guoh3dfeats'
    ckpt_name = 'last'
    cfg = read_config(run_dir)
    seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text_model = instantiate(cfg.data.text_to_token_emb, device=device)
    retrieval_model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    ##### ------ warm-up ------- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    for nb_iter in range(1, args.warm_up_iter):
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

        gt_motion, text = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()  # (bs, seq_len, dim)

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
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            logger.info(
                f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t "
                f"PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
            )

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net, logger, writer, 0,
            best_fid=1000, best_iter=0, best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper
        )

    for nb_iter in range(1, args.total_iter + 1):
        gt_motion, text = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()

        pred_motion, loss_commit, perplexity, projected_features = net(gt_motion)

        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        # 这里示例用了系数2做加权，你可根据需要自定义
        loss = 3 * loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        # 计算相似度并选正负样本（不变）
        motions_list = [m.cpu().numpy() for m in gt_motion]
        score_matrix = text_motion_sim_V1(text, motions_list, text_model, retrieval_model, normalizer, device=device)
        np.fill_diagonal(score_matrix, -np.inf)

        B = score_matrix.shape[0]
        x = args.x_pos_neg
        score_matrix_t = torch.from_numpy(score_matrix).float().cuda()
        top_values, top_indices = torch.topk(score_matrix_t, k=x, dim=1)
        bottom_values, bottom_indices = torch.topk(score_matrix_t, k=x, dim=1, largest=False)

        # 对比损失使用 projected_features
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

        # 打印训练过程
        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            writer.add_scalar('./Train/ContrastiveLoss', contrastive_loss_total.item(), nb_iter)

            logger.info(
                f"Train. Iter {nb_iter} : Commit. {avg_commit:.5f}, PPL. {avg_perplexity:.2f}, "
                f"Recons. {avg_recons:.5f}, Contrastive: {10*contrastive_loss_total.item():.5f}"
            )

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

        # 定期做评估
        if nb_iter % args.eval_iter == 0:
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
                eval_trans.evaluation_vqvae(
                    args.out_dir, val_loader, net, logger, writer, nb_iter,
                    best_fid, best_iter, best_div, best_top1,
                    best_top2, best_top3, best_matching,
                    eval_wrapper=eval_wrapper
                )
