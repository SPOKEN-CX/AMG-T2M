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


def weighted_info_nce_loss_batched(anchors, pos_samples, neg_samples, pos_weights, temperature):
    anchors_norm = torch.nn.functional.normalize(anchors, p=2, dim=1)
    pos_norm = torch.nn.functional.normalize(pos_samples, p=2, dim=2)
    neg_norm = torch.nn.functional.normalize(neg_samples, p=2, dim=2)
    pos_sim = torch.sum(anchors_norm.unsqueeze(1) * pos_norm, dim=2)
    neg_sim = torch.sum(anchors_norm.unsqueeze(1) * neg_norm, dim=2)
    numerator = torch.sum(pos_weights * torch.exp(pos_sim / temperature), dim=1)
    denominator = numerator + torch.sum(torch.exp(neg_sim / temperature), dim=1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()


if __name__ == '__main__':
    args = option_vq.get_args_parser()
    args.check_dir = './checkpoints_model/'
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)
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
        test_dataset = dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file, w_vectorizer)
        val_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True, shuffle=True, collate_fn=dataset_TM_eval.collate_fn)
    else:
        val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2 ** args.down_t)

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

    start_iter = 0

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

        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info("Scheduler state loaded from checkpoint.")

        start_iter = ckpt.get('iter', 0)
        logger.info(f"Resumed from iter {start_iter}")

    net.train()
    net.cuda()
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)
    args.x_pos_neg = 100
    args.contrastive_weight = 0.01
    args.contrastive_temp = 0.07

    if not args.resume_pth:
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        for nb_iter in range(1, args.warm_up_iter + 1):
            optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
            gt_motion, text, motion_retrival, text_retrival = next(train_loader_iter)
            gt_motion = gt_motion.cuda().float()
            pred_motion, loss_commit, perplexity, quantized = net(gt_motion)
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
                logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
        args.out_dir, val_loader, net, logger, writer, start_iter,
        best_fid=1000, best_iter=0, best_div=100, best_top1=0,
        best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper
    )

    for nb_iter in range(start_iter + 1, args.total_iter + 1):
        total_start = time.time()
        gt_motion, text, motion_retrival, text_retrival = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()
        pred_motion, loss_commit, perplexity, quantized = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = 4 * loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        text_retrival = text_retrival.squeeze(1)
        motion_retrival = motion_retrival.squeeze(1)
        score_matrix = text_motion_sim_V3(text_retrival, motion_retrival)
        B = score_matrix.size(0)
        indices = torch.arange(B, device=score_matrix.device)
        score_matrix[indices, indices] = float('-inf')
        x = args.x_pos_neg
        top_values, top_indices = torch.topk(score_matrix, k=x, dim=1)
        bottom_values, bottom_indices = torch.topk(score_matrix, k=x, dim=1, largest=False)
        anchors = quantized
        con_start = time.time()
        pos_samples = quantized[top_indices]
        neg_samples = quantized[bottom_indices]
        pos_scores = torch.tensor(top_values).to("cuda")
        pos_weights = torch.nn.functional.softmax(pos_scores / args.contrastive_temp, dim=1)
        con_end = time.time()
        contrastive_loss = weighted_info_nce_loss_batched(
            anchors, pos_samples, neg_samples, pos_weights, args.contrastive_temp
        )
        loss += args.contrastive_weight * contrastive_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            writer.add_scalar('./Train/ContrastiveLoss', contrastive_loss.item(), nb_iter)
            logger.info(f"Train. Iter {nb_iter} : Commit. {avg_commit:.5f}, PPL. {avg_perplexity:.2f}, Recons. {avg_recons:.5f}, Contrastive: {contrastive_loss.item():.5f}")
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

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
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
                args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter,
                best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper
            )
