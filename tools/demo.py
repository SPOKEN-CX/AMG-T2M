# -*- coding: utf-8 -*-
"""使用本地模型生成分段动作和 GIF；无需桌面环境，可从任意目录运行。"""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import argparse
from pathlib import Path
import re
from textwrap import wrap

import clip
import imageio.v2 as imageio
import matplotlib

# 必须在导入 pyplot 前选择服务器使用的无窗口后端。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

import models.vqvae as vqvae
import models.t2m_trans as trans
from utils.motion_process import recover_from_ric

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 支持字符串，或 {"text": "...", "n_tokens": 30}；按 step 数字顺序生成。
steps = {
    "step1": "a person waves their right hand",
    "step2": "a person walks sideways to the left",
    "step3": "a person raises both arms above their head",
}
DEFAULT_TOKENS_PER_STEP = 51
BLOCK_SIZE = 51


def project_path(value):
    """相对路径统一相对于项目目录，绝对路径保持不变。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-pth', type=project_path,
                        default=PROJECT_ROOT / 'checkpoints/self_train/VQVAE/t2m/net_best_fid.pth')
    parser.add_argument('--resume-trans', type=project_path,
                        default=PROJECT_ROOT / 'checkpoints/self_train/GPT/net_best_fid.pth')
    parser.add_argument('--meta-dir', type=project_path,
                        default=PROJECT_ROOT / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
    parser.add_argument('--clip-model', type=project_path,
                        default=PROJECT_ROOT / 'ViT-B-32.pt')
    parser.add_argument('--out-dir', type=project_path,
                        default=PROJECT_ROOT / 'generation/motion_gen')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def _step_sort_key(key):
    match = re.search(r'(\d+)', key)
    return (0, int(match.group(1))) if match else (1, key)


def normalize_steps(steps_dict):
    normalized = []
    for name, value in sorted(steps_dict.items(), key=lambda item: _step_sort_key(item[0])):
        if isinstance(value, str):
            text, n_tokens = value, DEFAULT_TOKENS_PER_STEP
        elif isinstance(value, dict):
            text = value.get('text', '')
            n_tokens = int(value.get('n_tokens', DEFAULT_TOKENS_PER_STEP))
        else:
            raise TypeError(f"steps[{name!r}] 只支持 str 或 dict")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{name}: 动作描述不能为空')
        if not 1 <= n_tokens <= BLOCK_SIZE:
            raise ValueError(f'{name}: n_tokens 必须在 1 到 {BLOCK_SIZE} 之间')
        normalized.append((name, text, n_tokens))
    if not normalized:
        raise ValueError('steps 不能为空')
    return normalized


def save_motion_gif(joints, caption, output_path, fps=20):
    """兼容新版 Matplotlib，直接写 GIF，避免堆叠全部 RGBA 帧为 tensor。"""
    data = joints.copy()
    data[:, :, 1] -= data[:, :, 1].min()
    trajectory = data[:, 0, [0, 2]].copy()
    data[:, :, 0] -= trajectory[:, 0, None]
    data[:, :, 2] -= trajectory[:, 1, None]
    chains = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
              [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
              [9, 13, 16, 18, 20]]
    colors = ['red', 'blue', 'black', 'red', 'blue']
    radius = max(1.5, float(np.abs(data[:, :, [0, 2]]).max()) * 1.1)
    height = max(2.0, float(data[:, :, 1].max()) * 1.1)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('\n'.join(wrap(caption, 60)), fontsize=11)
    try:
        with imageio.get_writer(str(output_path), mode='I', duration=1000 / fps, loop=0) as writer:
            for index, frame in enumerate(data):
                # ax.lines/collections 在新版 Matplotlib 中是只读属性。
                ax.clear()
                ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(0, height))
                ax.set_box_aspect((2 * radius, 2 * radius, height))
                ax.view_init(elev=20, azim=-70)
                ax.set_axis_off()
                # HumanML3D 的 Y 轴向上，映射到绘图坐标的 Z 轴。
                floor = [[-radius, -radius, 0], [-radius, radius, 0],
                         [radius, radius, 0], [radius, -radius, 0]]
                ax.add_collection3d(Poly3DCollection([floor], facecolors=(0.5, 0.5, 0.5, 0.2)))
                if index > 1:
                    trail = trajectory[:index] - trajectory[index]
                    ax.plot(trail[:, 0], trail[:, 1], np.zeros(index), color='blue', linewidth=1)
                for chain, color in zip(chains, colors):
                    ax.plot(frame[chain, 0], frame[chain, 2], frame[chain, 1],
                            color=color, linewidth=3)
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
                if (index + 1) % 100 == 0 or index == len(data) - 1:
                    print(f'Rendering GIF: {index + 1}/{len(data)} frames', flush=True)
    finally:
        plt.close(fig)


def main():
    args = parse_args()
    steps_list = normalize_steps(steps)
    required_files = [args.resume_pth, args.resume_trans, args.clip_model,
                      args.meta_dir / 'mean.npy', args.meta_dir / 'std.npy']
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f'缺少本地资源文件: {path}')
    # 项目中的 EMA quantizer 在初始化时调用 .cuda()，因此需要 CUDA。
    if not torch.cuda.is_available():
        raise RuntimeError('此项目的 VQVAE 需要 CUDA，请使用已配置的 t2mgpt conda 环境和 GPU。')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'Output directory: {args.out_dir}', flush=True)

    # 与原脚本及 self_train 权重一致的网络参数。
    model_args = argparse.Namespace(dataname='t2m', quantizer='ema_reset', mu=0.99)
    net = vqvae.HumanVQVAE(model_args, 512, 512, 512, 2, 2, 512, 3, 3)
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=512, embed_dim=1024, clip_dim=512, block_size=BLOCK_SIZE,
        num_layers=9, n_head=16, drop_out_rate=0.1, fc_rate=4)

    print(f'Loading VQVAE: {args.resume_pth}', flush=True)
    checkpoint = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=True)
    del checkpoint
    net.eval().to(device)
    print(f'Loading transformer: {args.resume_trans}', flush=True)
    checkpoint = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(checkpoint['trans'], strict=True)
    del checkpoint
    trans_encoder.eval().to(device)

    # 直接使用服务器已有的 CLIP 权重，不触发联网下载。
    print(f'Loading CLIP: {args.clip_model}', flush=True)
    clip_model, _ = clip.load(str(args.clip_model), device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval().requires_grad_(False)
    mean = torch.from_numpy(np.load(args.meta_dir / 'mean.npy')).to(device)
    std = torch.from_numpy(np.load(args.meta_dir / 'std.npy')).to(device)

    all_indices, all_texts = [], []
    with torch.no_grad():
        for name, text, n_tokens in steps_list:
            all_texts.append(text)
            text_tokens = clip.tokenize([text], truncate=True).to(device)
            text_features = clip_model.encode_text(text_tokens).float()
            # 当前 sampler 最多返回 block_size-1 个 token，也可能提前输出 EOS。
            try:
                indices = trans_encoder.sample(text_features, False)[:, :n_tokens]
            except UnboundLocalError as exc:
                raise RuntimeError(f'{name}: 模型首个 token 就输出 EOS，未生成动作，请调整描述。') from exc
            if indices.shape[1] == 0:
                raise RuntimeError(f'{name}: 未生成有效动作 token，请调整描述。')
            all_indices.append(indices)
            print(f'{name}: {text} -> {indices.shape[1]} tokens', flush=True)
        full_indices = torch.cat(all_indices, dim=1)
        pred_pose = net.forward_decoder(full_indices)
        xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
        xyz = xyz.reshape(1, -1, 22, 3).cpu().numpy()

    if not np.isfinite(xyz).all():
        raise RuntimeError('生成的动作包含 NaN 或 Inf，无法可视化。')
    motion_path = args.out_dir / 'motion.npy'
    gif_path = args.out_dir / 'example.gif'
    np.save(motion_path, xyz)
    print(f'Segment token lengths: {[item.shape[1] for item in all_indices]}', flush=True)
    print(f'Motion shape: {xyz.shape}', flush=True)
    save_motion_gif(xyz[0], ' -> '.join(all_texts), gif_path)
    print(f'Saved: {motion_path}\nSaved: {gif_path}', flush=True)


if __name__ == '__main__':
    main()
