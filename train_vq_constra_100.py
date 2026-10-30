import argparse
import os
import json
import warnings
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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

from dataset.dataset_amg_vq import AMGPairedVQDataset
from utils.amg_checkpoint import save_training_state, load_training_state

warnings.simplefilter("ignore", FutureWarning)


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


def weighted_info_nce_loss_batched(anchors, pos_samples, neg_samples, pos_weights, temperature):
    if temperature <= 0:
        raise ValueError('contrastive temperature must be positive')
    anchors_norm = torch.nn.functional.normalize(anchors, p=2, dim=1)
    pos_norm = torch.nn.functional.normalize(pos_samples, p=2, dim=2)
    neg_norm = torch.nn.functional.normalize(neg_samples, p=2, dim=2)
    pos_sim = torch.sum(anchors_norm.unsqueeze(1) * pos_norm, dim=2)
    neg_sim = torch.sum(anchors_norm.unsqueeze(1) * neg_norm, dim=2)
    log_num = torch.logsumexp(pos_weights.log() + pos_sim / temperature, dim=1)
    log_neg = torch.logsumexp(neg_sim / temperature, dim=1)
    loss = torch.logaddexp(log_num, log_neg) - log_num
    return loss.mean()


def select_neighbors(scores, k):
    batch_size = scores.shape[0]
    if scores.ndim != 2 or scores.shape[1] != batch_size or not torch.isfinite(scores).all():
        raise ValueError('Expected a finite square similarity matrix')
    if k < 1 or 2 * k > batch_size - 1:
        raise ValueError(f'Need 2*k <= batch_size-1, got k={k}, batch={batch_size}')
    positives = scores.clone()
    negatives = scores.clone()
    positives.fill_diagonal_(float('-inf'))
    negatives.fill_diagonal_(float('inf'))
    pos_scores, pos_idx = positives.topk(k, dim=1)
    negatives.scatter_(1, pos_idx, float('inf'))
    _, neg_idx = negatives.topk(k, dim=1, largest=False)
    return pos_scores, pos_idx, neg_idx


if __name__ == '__main__':
    args = option_vq.get_args_parser()
    args.check_dir = './checkpoints_model/'
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
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

    if args.dataname != 't2m':
        raise ValueError('This experimental paired training entry currently supports HumanML3D only')
    if args.amg_neighbors < 1 or 2 * args.amg_neighbors > args.batch_size - 1:
        raise ValueError('Require 1 <= 2 * --amg-neighbors <= --batch-size - 1')
    if args.amg_temperature <= 0 or args.amg_contrastive_weight < 0:
        raise ValueError('Invalid contrastive temperature or weight')
    train_set = AMGPairedVQDataset(args.amg_pairs_dir, window_size=args.window_size)
    if len(train_set) < args.batch_size:
        raise ValueError(f'Not enough training motions for batch size {args.batch_size}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
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
        val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer, unit_length=2 ** args.down_t)

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
    net.cuda()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    start_iter = 0
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100
    if args.resume_pth:
        start_iter, best = load_training_state(args.resume_pth, net, optimizer, scheduler, args)
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = best
        if start_iter > args.total_iter:
            raise ValueError('Checkpoint iteration exceeds total-iter')
        logger.info('Resumed model, optimizer, scheduler, EMA and RNG state at iteration %d; data iterator restarts', start_iter)

    net.train()
    net.cuda()
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)
    args.x_pos_neg = args.amg_neighbors
    args.contrastive_weight = args.amg_contrastive_weight
    args.contrastive_temp = args.amg_temperature

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
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite warmup loss at {nb_iter}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float('inf'), error_if_nonfinite=True)
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
    if not args.resume_pth:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
            args.out_dir, val_loader, net, logger, writer, start_iter,
            best_fid=1000, best_iter=0, best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False
        )
        save_training_state(os.path.join(args.out_dir, 'training_last.pth'), net, optimizer, scheduler,
                            start_iter, args, [best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching])

    for nb_iter in range(start_iter + 1, args.total_iter + 1):
        gt_motion, text, motion_retrival, text_retrival = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()
        pred_motion, loss_commit, perplexity, quantized = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        loss = 4 * loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        text_retrival = text_retrival.cuda().float()
        motion_retrival = motion_retrival.cuda().float()
        score_matrix = (F.normalize(text_retrival, dim=-1) @ F.normalize(motion_retrival, dim=-1).T + 1) / 2
        score_matrix = torch.round(score_matrix * 1000) / 1000
        top_values, top_indices, bottom_indices = select_neighbors(score_matrix, args.x_pos_neg)
        anchors = quantized
        pos_samples = quantized[top_indices]
        neg_samples = quantized[bottom_indices]
        pos_scores = top_values
        pos_weights = torch.nn.functional.softmax(pos_scores / args.contrastive_temp, dim=1)
        contrastive_loss = weighted_info_nce_loss_batched(
            anchors, pos_samples, neg_samples, pos_weights, args.contrastive_temp
        )
        loss += args.contrastive_weight * contrastive_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite AMG loss at iteration {nb_iter}')
        if nb_iter == start_iter + 1:
            encoder_params = tuple(net.vqvae.encoder.parameters())
            contrastive_grads = torch.autograd.grad(contrastive_loss, encoder_params, retain_graph=True, allow_unused=True)
            used_grads = [g for g in contrastive_grads if g is not None]
            if not used_grads or not all(torch.isfinite(g).all() for g in used_grads):
                raise FloatingPointError('Invalid contrastive gradients in VQ encoder')
            contrastive_grad_norm = torch.stack([g.norm() for g in used_grads]).norm().item()
            if contrastive_grad_norm == 0:
                raise RuntimeError('Contrastive loss has zero gradient in VQ encoder')
            logger.info('Contrastive encoder gradient norm: %.8f', contrastive_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float('inf'), error_if_nonfinite=True).item()
        optimizer.step()
        scheduler.step()
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            writer.add_scalar('./Train/TotalLoss', loss.item(), nb_iter)
            writer.add_scalar('./Train/GradNorm', grad_norm, nb_iter)
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            writer.add_scalar('./Train/ContrastiveLoss', contrastive_loss.item(), nb_iter)
            logger.info(f"Train. Iter {nb_iter} : Commit. {avg_commit:.5f}, PPL. {avg_perplexity:.2f}, Recons. {avg_recons:.5f}, Contrastive: {contrastive_loss.item():.5f}, Total(last): {loss.item():.5f}, GradNorm(last): {grad_norm:.5f}")
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

        if nb_iter % args.eval_iter == 0:
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(
                args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter,
                best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper, draw=False
            )

        if nb_iter % args.save_iter == 0:
            best = [best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching]
            save_training_state(os.path.join(args.out_dir, 'training_last.pth'), net, optimizer, scheduler, nb_iter, args, best)
            checkpoint_dir = os.path.join(args.check_dir, args.exp_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_training_state(os.path.join(checkpoint_dir, f'model_{nb_iter}.pth'), net, optimizer, scheduler, nb_iter, args, best)
            logger.info('Saved training state including EMA at iteration %d', nb_iter)

    torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, 'net_last.pth'))
    save_training_state(os.path.join(args.out_dir, 'training_last.pth'), net, optimizer, scheduler,
                        args.total_iter, args, [best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching])
    logger.info('Training completed; saved final model and optimizer/scheduler checkpoint at iteration %d', args.total_iter)
    writer.flush()
    writer.close()
