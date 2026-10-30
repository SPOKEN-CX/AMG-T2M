
import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import os
import itertools
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from datasets import dataset_TM_train
from datasets import dataset_TM_eval
from datasets import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from utils.word_vectorizer import WordVectorizer

warnings.filterwarnings('ignore')

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
if args.bf16 and not torch.cuda.is_bf16_supported():
    raise RuntimeError('BF16 is unsupported')
if args.save_iter < 1 or args.grad_accum_steps < 1:
    raise ValueError('--save-iter and --grad-accum-steps must be positive')
if args.validation_max_batches != 0 and args.validation_max_batches < 4:
    raise ValueError('--validation-max-batches must be 0 or at least 4')
args.effective_batch_size = args.batch_size * args.grad_accum_steps

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_data_root = ('../Dataset/BABEL_TEACH' if args.dataname == 'BABEL_TEACH'
                     else os.path.join(str(_PROJECT_ROOT), 'dataset', 'HumanML3D'))
args.vq_dir = os.path.join(args.vq_data_root,
                           f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.vq_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2 ** args.down_t)
if len(train_loader_token.dataset) == 0:
    raise RuntimeError(f'No training motions found for {args.dataname}')
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
    dataset_opt_path = './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    split_file = pjoin(opt.data_root, 'test_2000.txt')
    test_dataset = dataset_TM_eval.st2m_Text2Motion_withpast_Dataset_evalV2(
        opt, mean, std, split_file, w_vectorizer,
    )
    val_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True, shuffle=True,
                            collate_fn=dataset_TM_eval.collate_fn)
    print('BABEL text_dataloader successfully loaded!')

else:

    from utils.word_vectorizer import WordVectorizer

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval.DATALoader(
        args.dataname, False, 32, w_vectorizer,
        unit_length=2**args.down_t,
    )

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

if args.validation_max_batches:
    class LimitedValidationLoader:
        def __init__(self, loader, limit):
            self.loader, self.limit, self.dataset = loader, min(limit, len(loader)), loader.dataset

        def __iter__(self):
            return itertools.islice(self.loader, self.limit)

        def __len__(self):
            return self.limit

    val_loader = LimitedValidationLoader(val_loader, args.validation_max_batches)
    logger.info('Validation restricted to %d batches for smoke checking', len(val_loader))

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load('./ViT-B-32.pt', device=device, jit=False)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
net = vqvae.HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       )

trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                              embed_dim=args.embed_dim_gpt,
                                              clip_dim=args.clip_dim,
                                              block_size=args.block_size,
                                              num_layers=args.num_layers,
                                              n_head=args.n_head_gpt,
                                              drop_out_rate=args.drop_out_rate,
                                              fc_rate=args.ff_rate)

clip_version = 'ViT-B/32'

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cuda:0')
net.load_state_dict(ckpt['net'], strict=True)
print('VQVAE loaded successfully.')
net.eval()
net.to(device)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

trans_encoder.train()
trans_encoder.to(device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

##### ---- get code ---- #####
net.eval()
with torch.no_grad():
    for batch in train_loader_token:
        pose, name = batch
        pose = pose.to(device).float()
        target = net.encode(pose).cpu().numpy()
        np.save(pjoin(args.vq_dir, name[0] + '.npy'), target)

train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name,
                                           unit_length=2 ** args.down_t)
if len(train_loader.dataset) < args.batch_size:
    raise RuntimeError(
        f'有效 GPT 训练样本不足一个 batch: {len(train_loader.dataset)} < {args.batch_size}'
    )
train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Training ---- #####
best_fid, best_iter, best_div = 1000., 0, 100.
best_top1 = best_top2 = best_top3 = 0.
best_matching = 100.
if args.resume_training:
    state = torch.load(args.resume_training, map_location='cpu')
    if state['args']['resume_pth'] != args.resume_pth or state['args']['vq_name'] != args.vq_name:
        raise ValueError('Full resume requires the same VQ path and token directory')
    if (state['args']['batch_size'] != args.batch_size or
            state['args'].get('grad_accum_steps', 1) != args.grad_accum_steps):
        raise ValueError('Full resume requires the same microbatch and accumulation settings')
    trans_encoder.load_state_dict(state['trans'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    nb_iter = state['iter']
    (best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching) = state['best']
    random.setstate(state['random_state'])
    np.random.set_state(state['numpy_state'])
    torch.set_rng_state(state['torch_state'])
    torch.cuda.set_rng_state_all(state['cuda_states'])
    logger.info('Restored iteration %d; DataLoader order restarts', nb_iter)
elif not args.skip_initial_eval:
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(
        args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=best_fid,
        best_iter=best_iter, best_div=best_div, best_top1=best_top1, best_top2=best_top2,
        best_top3=best_top3, best_matching=best_matching, clip_model=clip_model,
        eval_wrapper=eval_wrapper, draw=False)

def save_training_state():
    checkpoint = {
        'trans': trans_encoder.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'iter': nb_iter, 'args': vars(args),
        'best': (best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching),
        'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(), 'cuda_states': torch.cuda.get_rng_state_all(),
    }
    tmp = os.path.join(args.out_dir, 'training_last.pth.tmp')
    torch.save(checkpoint, tmp)
    os.replace(tmp, os.path.join(args.out_dir, 'training_last.pth'))
    writer.flush()
    logger.info('Saved full training state at iteration %d', nb_iter)

def train_microbatch(batch):
    clip_text, m_tokens, m_tokens_len, total_step, text_1 = batch
    m_tokens, m_tokens_len = m_tokens.to(device), m_tokens_len.to(device)
    target = m_tokens

    text = clip.tokenize(clip_text, truncate=True).to(device)
    feat_clip_text = clip_model.encode_text(text).float()  # shape: (bs, 512)

    input_index = target[:, :-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape, device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape, device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)

    # print('r_index', r_indices[0])
    a_indices = mask * input_index + (1 - mask) * r_indices

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        cls_pred = trans_encoder(a_indices, feat_clip_text)
        positions = torch.arange(target.shape[1], device=device)[None, :]
        valid = positions < (m_tokens_len[:, None] + 1)
        token_loss = torch.nn.functional.cross_entropy(
            cls_pred.float().transpose(1, 2), target, reduction='none',
            ignore_index=args.nb_code + 1)
        loss_cls = ((token_loss * valid).sum(1) / valid.sum(1)).mean()
    with torch.no_grad():
        if args.if_maxtest:
            predicted = cls_pred.argmax(dim=-1)
        else:
            predicted = Categorical(logits=cls_pred.float()).sample()
        correct = ((predicted == target) & valid).sum().item()
        tokens = (m_tokens_len + 1).sum().item()
    return loss_cls, correct, tokens


logger.info('GPT microbatch %d, accumulation %d, effective batch %d; iterations count optimizer updates',
            args.batch_size, args.grad_accum_steps, args.effective_batch_size)
while nb_iter < args.total_iter:
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.grad_accum_steps):
        loss_cls, correct, tokens = train_microbatch(next(train_loader_iter))
        if not torch.isfinite(loss_cls):
            raise RuntimeError(f'Nonfinite loss at iteration {nb_iter}')
        (loss_cls / args.grad_accum_steps).backward()
        avg_loss_cls += loss_cls.item() / args.grad_accum_steps
        right_num += correct
        nb_sample_train += tokens
    optimizer.step()
    scheduler.step()

    nb_iter += 1
    if nb_iter % args.print_iter == 0:
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
            eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter,
                                              best_fid, best_iter, best_div, best_top1, best_top2, best_top3,
                                              best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False)

    if nb_iter == 20 or nb_iter % args.save_iter == 0 or nb_iter == args.total_iter:
        save_training_state()

    if nb_iter == args.total_iter:
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break

writer.close()
