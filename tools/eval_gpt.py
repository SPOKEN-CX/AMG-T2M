
import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from datasets import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

if not args.resume_pth or not os.path.isfile(args.resume_pth):
    raise FileNotFoundError(f'VQ checkpoint is missing: {args.resume_pth!r}')
if not args.resume_trans or not os.path.isfile(args.resume_trans):
    raise FileNotFoundError(f'GPT checkpoint is missing: {args.resume_trans!r}')
if not os.path.isfile('./ViT-B-32.pt'):
    raise FileNotFoundError('Required local CLIP weights are missing: ./ViT-B-32.pt')

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model_path = './ViT-B-32.pt'
clip_model, clip_preprocess = clip.load(clip_model_path, device=torch.device('cuda'), jit=False)
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cuda:0')
incompatible = net.load_state_dict(ckpt['net'], strict=False)
expected_missing = {'projection_layer.weight', 'projection_layer.bias'}
missing_keys = set(incompatible.missing_keys)
if missing_keys not in (set(), expected_missing) or incompatible.unexpected_keys:
    raise RuntimeError(f'Unexpected VQ-VAE checkpoint mismatch: {incompatible}')
net.eval()
net.cuda()
print ('loading VQ-VAE successful !')

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cuda:0')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

print ('loading VQ-Transformer successful !')


fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = args.repeat_time
if repeat_time < 1:
    raise ValueError('--repeat-time must be at least 1')

        
for i in range(repeat_time):
    result = eval_trans.evaluation_transformer_test(
        args.out_dir, val_loader, net, trans_encoder, logger, writer, 0,
        best_fid=1000, best_iter=0, best_div=100,
        best_top1=0, best_top2=0, best_top3=0,
        best_matching=100, best_multi=0,
        clip_model=clip_model, eval_wrapper=eval_wrapper,
        draw=False, save=False, savegif=False, savenpy=False,
    )
    fid_i, _, div_i, top1_i, top2_i, top3_i, matching_i, multi_i, writer, logger = result
    fid.append(fid_i)
    div.append(div_i)
    top1.append(top1_i)
    top2.append(top2_i)
    top3.append(top3_i)
    matching.append(matching_i)
    multi.append(multi_i)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
print('multimodality: ', sum(multi)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multimodality. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)
