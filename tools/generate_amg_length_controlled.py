#!/usr/bin/env python3
"""Generate one motion per caption using a predicted and enforced length."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import clip
from models.t2m_trans import Text2Motion_Transformer
from models.vqvae import HumanVQVAE
from utils.motion_process import recover_from_ric


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frames_to_tokens(frames, stride=4, max_tokens=49, allow_clip=False):
    if not math.isfinite(frames) or frames <= 0:
        raise ValueError('Predicted frames must be positive and finite')
    limit = stride * max_tokens
    if frames > limit and not allow_clip:
        raise ValueError(f'Predicted length {frames:.2f} frames exceeds the {limit}-frame GPT limit; split the text or explicitly use --allow-clip')
    tokens = max(1, math.floor(frames / stride + 0.5))
    return min(tokens, max_tokens)


@torch.inference_mode()
def sample_exact_tokens(model, clip_feature, count, categorical=False):
    if clip_feature.shape[0] != 1 or not 1 <= count <= model.block_size - 2:
        raise ValueError('One caption and between 1 and block_size - 2 tokens are required')
    tokens = torch.empty((1, 0), dtype=torch.long, device=clip_feature.device)
    for _ in range(count):
        logits = model(tokens, clip_feature)[:, -1, :].clone()
        logits[:, model.num_vq] = -torch.inf
        if not torch.isfinite(logits[:, :model.num_vq]).all():
            raise ValueError('Nonfinite GPT motion-token logits')
        next_token = (torch.distributions.Categorical(logits=logits).sample()[:, None]
                      if categorical else logits.argmax(dim=-1, keepdim=True))
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpt-checkpoint', type=Path, required=True)
    p.add_argument('--vq-checkpoint', type=Path, default=ROOT/'output/hml3d/VQ_amg_b512_frozen/net_best_fid.pth')
    p.add_argument('--length-checkpoint', type=Path, default=ROOT/'output/hml3d/Length_amg_gat_b512_50k/net_best_mae.pth')
    p.add_argument('--spacy-model', default='en_core_web_sm')
    p.add_argument('--text', nargs='+', required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--categorical', action='store_true', help='sample motion tokens rather than choose the most probable')
    p.add_argument('--allow-clip', action='store_true', help='explicitly clip predicted durations above 196 frames')
    args = p.parse_args()
    if (args.out_dir / 'metadata.json').exists():
        raise FileExistsError('Output already exists; choose a new directory')
    for path in (args.gpt_checkpoint, args.vq_checkpoint, args.length_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity_path = args.gpt_checkpoint.resolve().parent / 'vq_identity.json'
    if not identity_path.is_file():
        raise FileNotFoundError(f'GPT VQ identity file missing: {identity_path}')
    identity = json.loads(identity_path.read_text())
    vq_sha = sha256(args.vq_checkpoint)
    if vq_sha != identity['vq_sha256']:
        raise ValueError('GPT checkpoint and VQ checkpoint are not the verified pair')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory) / 'predicted_lengths.json'
        subprocess.run([sys.executable, str(ROOT/'tools/predict_amg_lengths.py'),
                        '--checkpoint', str(args.length_checkpoint), '--spacy-model', args.spacy_model,
                        '--text', *args.text, '--output', str(tmp)], check=True, cwd=ROOT)
        predictions = json.loads(tmp.read_text())
    counts = [frames_to_tokens(p['pred_frames'], allow_clip=args.allow_clip) for p in predictions]
    clip_model, _ = clip.load(str(ROOT/'ViT-B-32.pt'), device='cuda', jit=False)
    clip_model.eval().requires_grad_(False)
    model = Text2Motion_Transformer(num_vq=512, embed_dim=1024, clip_dim=512, block_size=51,
                                   num_layers=9, n_head=16, drop_out_rate=.1, fc_rate=4).cuda().eval()
    model.load_state_dict(torch.load(args.gpt_checkpoint, map_location='cpu')['trans'], strict=True)
    vq_args = argparse.Namespace(dataname='t2m', quantizer='ema_reset', mu=.99)
    vq = HumanVQVAE(vq_args, nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
                    stride_t=2, width=512, depth=3, dilation_growth_rate=3).cuda().eval()
    vq.load_state_dict(torch.load(args.vq_checkpoint, map_location='cpu')['net'], strict=True)
    meta = ROOT/'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    mean, std = np.load(meta/'mean.npy'), np.load(meta/'std.npy')
    records = []
    with torch.inference_mode():
        features = clip_model.encode_text(clip.tokenize(args.text, truncate=True).cuda()).float()
        for i, (prediction, count) in enumerate(zip(predictions, counts)):
            tokens = sample_exact_tokens(model, features[i:i+1], count, args.categorical)
            normalized = vq.forward_decoder(tokens)[0].cpu().numpy()
            if normalized.shape != (count * 4, 263) or not np.isfinite(normalized).all():
                raise RuntimeError('VQ decoded motion has invalid length, shape, or values')
            motion = normalized * std + mean
            xyz = recover_from_ric(torch.from_numpy(motion).float().unsqueeze(0), 22)[0].numpy()
            if not np.isfinite(xyz).all():
                raise RuntimeError('Nonfinite joint coordinates')
            prefix = f'clip_{i:03d}'
            np.save(args.out_dir/f'{prefix}_tokens.npy', tokens.cpu().numpy()[0])
            np.save(args.out_dir/f'{prefix}_motion.npy', motion)
            np.save(args.out_dir/f'{prefix}_joints.npy', xyz)
            records.append({**prediction, 'token_count': count, 'generated_frames': len(motion),
                            'clipped_to_limit': prediction['pred_frames'] > 196,
                            'tokens_file': f'{prefix}_tokens.npy', 'motion_file': f'{prefix}_motion.npy',
                            'joints_file': f'{prefix}_joints.npy'})
    metadata = {'records': records, 'seed': args.seed, 'sampling': 'categorical' if args.categorical else 'greedy',
                'length_policy': 'EOS suppressed until requested token count', 'fps': 20, 'frames_per_token': 4,
                'gpt_checkpoint': str(args.gpt_checkpoint.resolve()), 'gpt_sha256': sha256(args.gpt_checkpoint),
                'vq_checkpoint': str(args.vq_checkpoint.resolve()), 'vq_sha256': vq_sha,
                'length_checkpoint': str(args.length_checkpoint.resolve()), 'length_sha256': sha256(args.length_checkpoint),
                'mean_sha256': sha256(meta/'mean.npy'), 'std_sha256': sha256(meta/'std.npy')}
    (args.out_dir/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
