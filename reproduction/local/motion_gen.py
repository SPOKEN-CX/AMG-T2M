# -*- coding: utf-8 -*-

import sys
sys.argv = ['GPT_eval_multi.py']

import re
import warnings
warnings.filterwarnings('ignore')

import options.option_transformer as option_trans
args = option_trans.get_args_parser()

# =========================
# 1) 用 steps 字典作为输入
# =========================
# value 支持两种写法：
#   A) "text prompt"
#   B) {"text": "...", "n_tokens": 51}  # 可为每段指定生成 token 数
steps = {
    "step1": "a person is jumping",
    "step2": "then the person walks forward",
    "step3": "finally the person sits down"
}

# 如果你不想每段单独指定长度，就用这个默认长度（通常= args.block_size 或 args.block_size-?）
DEFAULT_TOKENS_PER_STEP = 51

# =========================
# 2) 原来的参数/模型加载
# =========================
args.dataname = 't2m'
args.resume_pth = '/home/peng/PythonProjects/T2M-GPT-main/checkpoints/self_train/VQVAE/t2m/net_best_fid.pth'
args.resume_trans = '/home/peng/PythonProjects/T2M-GPT-main/checkpoints/self_train/GPT/net_best_fid.pth'
args.down_t = 2
args.depth = 3
args.block_size = 51  # 这里建议与 DEFAULT_TOKENS_PER_STEP 一致（或 >= 你每段最大 n_tokens）

import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans

device = torch.device('cuda')

# load clip model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root='./')
clip.model.convert_weights(clip_model)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# VQVAE
net = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate
)

# Transformer
trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=1024,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=9,
    n_head=16,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate
)

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval().to(device)

print('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval().to(device)

mean = torch.from_numpy(
    np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
).to(device)
std = torch.from_numpy(
    np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')
).to(device)

# =========================
# 3) 分段采样码本序列，并拼接
# =========================
def _step_sort_key(k: str):
    # step1/step2/... 按数字排序；没有数字就按字符串
    m = re.search(r'(\d+)', k)
    return int(m.group(1)) if m else k

def normalize_steps(steps_dict):
    items = list(steps_dict.items())
    # 如果你希望严格按 step 编号顺序，用下面这一行排序；
    # 如果你希望保持你写入 dict 的顺序，就注释掉下一行。
    items.sort(key=lambda kv: _step_sort_key(kv[0]))

    norm = []
    for k, v in items:
        if isinstance(v, str):
            norm.append((k, v, DEFAULT_TOKENS_PER_STEP))
        elif isinstance(v, dict):
            text = v.get("text", "")
            n_tokens = int(v.get("n_tokens", DEFAULT_TOKENS_PER_STEP))
            norm.append((k, text, n_tokens))
        else:
            raise TypeError(f"steps['{k}'] 只支持 str 或 dict，当前是: {type(v)}")
    return norm

steps_list = normalize_steps(steps)

all_indices = []
all_texts = []

with torch.no_grad():
    for step_name, step_text, n_tokens in steps_list:
        all_texts.append(step_text)

        # CLIP text feat
        text_tok = clip.tokenize([step_text], truncate=True).to(device)
        feat_clip_text = clip_model.encode_text(text_tok).float()

        # 采样一段码本 index 序列（长度通常为 args.block_size 或与实现有关）
        idx = trans_encoder.sample(feat_clip_text[0:1], False)  # idx shape: [1, L]
        # 截断成你想要的分段长度
        if n_tokens is not None:
            idx = idx[:, :n_tokens]

        all_indices.append(idx)

# 合并为完整码本序列 [1, sum(L_i)]
full_index_motion = torch.cat(all_indices, dim=1)

print("Segment token lengths:", [x.shape[1] for x in all_indices])
print("Full token length:", full_index_motion.shape[1])

# =========================
# 4) 统一解码 → xyz → 保存 → 画一个 gif
# =========================
with torch.no_grad():
    pred_pose = net.forward_decoder(full_index_motion)  # [1, T, ...] 具体看实现

from utils.motion_process import recover_from_ric
pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
xyz = pred_xyz.reshape(1, -1, 22, 3)

np.save('motion.npy', xyz.detach().cpu().numpy())

import visualization.plot_3d_global as plot_3d
# gif 的标题：把所有 step 文本拼起来（你也可以改成 step 名称）
caption = " -> ".join(all_texts)
pose_vis = plot_3d.draw_to_batch(
    xyz.detach().cpu().numpy(),
    [caption],                # batch=1
    ['example.gif']           # 输出一个 gif
)

print("Saved: motion.npy, example.gif")
