import os
import json
import argparse


import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from os.path import join as pjoin
from torch.utils.data import DataLoader
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit':
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
elif args.dataname == 't2m':
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
elif args.dataname == 'BABEL_TEACH':
    dataset_opt_path = './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt'
    args.nb_joints = 22

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
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
else:
    args.nb_joints = 21 if args.dataname == 'kit' else 22

    val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2 ** args.down_t)
print(f' args.width :{args.width}\n args.depth: {args.depth}\n args.down_t : {args.down_t}\nargs.vq_act:{args.vq_act}\nargs.vq_norm:{args.vq_norm}')

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
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cuda:0')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 20
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0))
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)