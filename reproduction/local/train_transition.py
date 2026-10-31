import os
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
from models.MotionLenEstimator.LengthEst import *
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
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

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir = os.path.join("../Dataset/BABEL_TEACH" if args.dataname == 'BABEL_TEACH' else "../Dataset/HumanML3D",
                           f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.vq_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2 ** args.down_t)
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
    val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 16, w_vectorizer)

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
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
for batch in train_loader_token:
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.to(device).float()  # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()
    # print(f'ars.vq_dir:{args.vq_dir}')
    np.save(pjoin(args.vq_dir, name[0] + '.npy'), target)

train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name,
                                           unit_length=2 ** args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Training ---- #####
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(
    args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100,
    best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)
while nb_iter <= args.total_iter:

    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, total_step, text_1 = batch
    steps_dict_list = [json.loads(x) for x in text_1]  # list of dict
    step = total_step[0]

    # for s in range(1,step+1):
    #     key="Step "+str(s)
    #     step_text=[d.get(key, "") for d in steps_dict_list]
    #     #print(f'step {s} info : {step_text} ')
    # #t = json.loads(text_1[0])
    m_tokens, m_tokens_len = m_tokens.to(device), m_tokens_len.to(device)
    bs = m_tokens.shape[0]
    target = m_tokens  # (bs, 26)
    target = target.to(device)

    text = clip.tokenize(clip_text, truncate=True).to(device)
    feat_clip_text = clip_model.encode_text(text).float()  # shape: (bs, 512)

    cap_len_tensor = torch.tensor([len(t.split()) for t in clip_text]).float().to(device).view(bs, 1)
    # total_step_tensor = total_step.view(bs, 1).float().to(device)

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

    cls_pred = trans_encoder(a_indices, feat_clip_text)
    # print('cls_pred', cls_pred[0])
    cls_pred = cls_pred.contiguous()

    loss_cls = 0.0
    for i in range(bs):
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs

        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)
        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)
        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()

    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls += loss_cls.item()
    nb_sample_train += (m_tokens_len + 1).sum().item()

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
                                              best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper)

    if nb_iter == args.total_iter:
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break