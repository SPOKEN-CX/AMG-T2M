"""Run local T2M-GPT demos without the full HumanML3D dataset.

Examples (from the repository root):
  .venv-smoke/bin/python scripts/run_local.py preview
  .venv-smoke/bin/python scripts/run_local.py reconstruct
  .venv-smoke/bin/python scripts/run_local.py train-smoke --steps 3
  .venv-smoke/bin/python scripts/run_local.py generate --text 'a person walks forward'
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
META = ROOT / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
VQ = ROOT / 'pretrained/VQVAE/net_best_fid.pth'
GPT = ROOT / 'pretrained/VQTransformer_corruption05/net_best_fid.pth'


def write_report(out, report):
    (out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def render(xyz, target, title, fps=20):
    """Modern Matplotlib + Pillow rendering; no SMPL or ffmpeg required."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from textwrap import fill

    data = np.asarray(xyz, dtype=np.float32)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3 or data.shape[1:] != (22, 3) or not len(data):
        raise ValueError(f'Expected (T,22,3) or (1,T,22,3), got {data.shape}')
    if not np.isfinite(data).all():
        raise ValueError('Motion contains non-finite coordinates')
    data = data.copy()
    data[..., 1] -= data[..., 1].min()
    data[..., 0] -= data[:, :1, 0]
    data[..., 2] -= data[:, :1, 2]
    chains = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
              [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(fill(title, 48), fontsize=10)
    extent = max(float(np.abs(data[..., [0, 2]]).max()) * 1.1, 1.0)
    height = max(float(data[..., 1].max()) * 1.1, 1.8)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(0, height))
    ax.set_box_aspect((2 * extent, 2 * extent, height))
    ax.view_init(elev=15, azim=-70)
    ax.set_axis_off()
    lines = [ax.plot([], [], [], color=c, linewidth=3)[0]
             for c in ['red', 'blue', 'black', 'red', 'blue']]

    def update(i):
        for line, chain in zip(lines, chains):
            points = data[i, chain]
            line.set_data_3d(points[:, 0], points[:, 2], points[:, 1])
        return lines

    animation = FuncAnimation(fig, update, frames=len(data), interval=1000 / fps)
    animation.save(str(target), writer=PillowWriter(fps=fps))
    plt.close(fig)


def setup_torch(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'PyTorch {torch.__version__}; device={device}', flush=True)
    return torch.device(device)


def load_vq(path, device):
    import torch
    from models.vqvae import HumanVQVAE
    state = torch.load(str(path), map_location='cpu', weights_only=True)['net']
    codes, dim = state['vqvae.quantizer.codebook'].shape
    args = SimpleNamespace(dataname='t2m', quantizer='ema_reset', mu=0.99)
    model = HumanVQVAE(args, nb_code=codes, code_dim=dim, output_emb_width=dim,
                      down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3)
    incompatible = model.load_state_dict(state, strict=False)
    # The official checkpoint predates this fork's unused projection layer.
    allowed = {'projection_layer.weight', 'projection_layer.bias'}
    if set(incompatible.missing_keys) - allowed or incompatible.unexpected_keys:
        raise RuntimeError(f'Incompatible VQ checkpoint: {incompatible}')
    if incompatible.missing_keys:
        print('Official VQ checkpoint: unused projection_layer has no saved weights.', flush=True)
    return model.to(device).eval()


def load_stats():
    mean = np.load(META / 'mean.npy').astype(np.float32)
    std = np.load(META / 'std.npy').astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,) or not (std > 0).all():
        raise ValueError('Invalid HumanML3D normalization statistics')
    return mean, std


def read_features(path):
    x = np.load(path, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != 263 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite raw (T,263) features at {path}, got {x.shape}')
    # VQ encoder downsamples by four.
    return x[:len(x) // 4 * 4].astype(np.float32)


def reconstruct(args, out):
    import torch
    from utils.motion_process import recover_from_ric
    device = setup_torch(args)
    model = load_vq(args.vq, device)
    mean, std = load_stats()
    raw = read_features(args.source or ROOT / 'motions/000090.npy')
    motion = torch.from_numpy((raw - mean) / std).unsqueeze(0).to(device)
    with torch.inference_mode():
        indices = model.encode(motion)
        prediction = model.forward_decoder(indices)
        features = prediction.cpu().numpy()[0] * std + mean
        xyz = recover_from_ric(torch.from_numpy(features).unsqueeze(0), 22).numpy()
        original_xyz = recover_from_ric(torch.from_numpy(raw).unsqueeze(0), 22).numpy()
    np.save(out / 'tokens.npy', indices.cpu().numpy())
    np.save(out / 'reconstruction_features.npy', features)
    np.save(out / 'reconstruction.npy', xyz)
    np.save(out / 'original.npy', original_xyz)
    if not args.no_gif:
        render(original_xyz, out / 'original.gif', 'Local motion sample (raw features)')
        render(xyz, out / 'reconstruction.gif', 'VQ-VAE reconstruction')
    write_report(out, {'mode': args.mode, 'source': str(args.source or ROOT / 'motions/000090.npy'),
                       'vq': str(args.vq), 'device': str(device), 'frames': len(raw),
                       'tokens': indices.shape[1], 'xyz_shape': list(xyz.shape),
                       'raw_feature_mae': float(np.abs(features - raw).mean()),
                       'purpose': 'Local reconstruction check; not a benchmark metric.'})


def train_smoke(args, out):
    import torch
    from utils.losses import ReConsLoss
    device = setup_torch(args)
    mean, std = load_stats()
    paths = sorted((args.source or ROOT / 'motions').glob('*.npy'))
    sequences = [read_features(p) for p in paths]
    sequences = [s for s in sequences if len(s) >= 64]
    if not sequences:
        raise ValueError('No (T,263) samples with at least 64 frames')
    model = load_vq(args.vq, device).train()
    quantizer = model.vqvae.quantizer
    # Resume EMA statistics from the saved codebook instead of resetting it.
    quantizer.init = True
    quantizer.code_sum = quantizer.codebook.clone()
    quantizer.code_count = torch.ones(quantizer.nb_code, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.99))
    loss_fn = ReConsLoss('l1_smooth', 22)
    history = []
    probe = next(model.vqvae.encoder.parameters())
    before = probe.detach().clone()
    for step in range(args.steps):
        windows = []
        for _ in range(2):
            sequence = random.choice(sequences)
            start = random.randint(0, len(sequence) - 64)
            windows.append((sequence[start:start + 64] - mean) / std)
        batch = torch.from_numpy(np.stack(windows)).to(device)
        prediction, commit, perplexity, _ = model(batch)
        reconstruction = loss_fn(prediction, batch)
        loss = reconstruction + 0.02 * commit + 0.5 * loss_fn.forward_vel(prediction, batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Non-finite gradients')
        optimizer.step()
        entry = {'step': step + 1, 'loss': loss.item(), 'reconstruction': reconstruction.item(),
                 'commitment': commit.item(), 'perplexity': perplexity.item()}
        history.append(entry)
        print(json.dumps(entry), flush=True)
    delta = (probe.detach() - before).abs().max().item()
    if delta <= 0:
        raise RuntimeError('Optimizer did not change encoder weights')
    torch.save({'net': model.state_dict(), 'smoke_only': True}, out / 'smoke_vq.pth')
    write_report(out, {'mode': args.mode, 'device': str(device), 'usable_sequences': len(sequences),
                       'encoder_max_weight_change': delta, 'history': history,
                       'purpose': 'Forward/backward/optimizer check on local samples; not a trained model or benchmark.'})


def generate(args, out):
    import torch
    import clip
    from models.t2m_trans import Text2Motion_Transformer
    from utils.motion_process import recover_from_ric
    device = setup_torch(args)
    model = load_vq(args.vq, device)
    state = torch.load(str(args.gpt), map_location='cpu', weights_only=True)['trans']
    size = state['trans_base.tok_emb.weight'].shape[1]
    block = state['trans_base.pos_embed.embed'].shape[0]
    layers = len({key.split('.')[2] for key in state if key.startswith('trans_base.blocks.')})
    codes = model.vqvae.num_code
    transformer = Text2Motion_Transformer(num_vq=codes, embed_dim=size, clip_dim=512,
                                         block_size=block, num_layers=layers, n_head=16,
                                         fc_rate=4, drop_out_rate=0.1)
    transformer.load_state_dict(state, strict=True)
    del state
    transformer.to(device).eval()
    # Explicit local filename prevents a redundant CLIP weight download.
    clip_model, _ = clip.load(str(ROOT / 'ViT-B-32.pt'), device='cpu', jit=False)
    clip_model.float().to(device).eval()
    mean, std = load_stats()
    segments = []
    texts = args.text or ['a person walks forward']
    with torch.inference_mode():
        for text in texts:
            condition = clip_model.encode_text(clip.tokenize([text], truncate=True).to(device)).float()
            tokens = None
            # An EOS prediction at step zero should produce a clear error, not
            # the unbound `xs` exception in the original sample() implementation.
            for _ in range(min(args.max_tokens, block - 1)):
                logits = transformer([] if tokens is None else tokens, condition)[:, -1]
                index = (torch.distributions.Categorical(logits=logits).sample().view(1, 1)
                         if args.sample else logits.argmax(-1, keepdim=True))
                if index.item() == codes:
                    break
                tokens = index if tokens is None else torch.cat([tokens, index], dim=1)
            if tokens is None:
                raise RuntimeError('GPT predicted EOS before any motion token; try another text or --sample.')
            segments.append(tokens)
            print(f'{text}: {tokens.shape[1]} tokens', flush=True)
        indices = torch.cat(segments, dim=1)
        features = model.forward_decoder(indices).cpu().numpy()[0] * std + mean
        xyz = recover_from_ric(torch.from_numpy(features).unsqueeze(0), 22).numpy()
    if not np.isfinite(xyz).all():
        raise RuntimeError('Generated motion contains non-finite coordinates')
    np.save(out / 'tokens.npy', indices.cpu().numpy())
    np.save(out / 'features.npy', features)
    np.save(out / 'motion.npy', xyz)
    if not args.no_gif:
        render(xyz, out / 'motion.gif', ' -> '.join(texts))
    write_report(out, {'mode': args.mode, 'device': str(device), 'texts': texts,
                       'vq': str(args.vq), 'gpt': str(args.gpt), 'seed': args.seed,
                       'segment_tokens': [s.shape[1] for s in segments],
                       'xyz_shape': list(xyz.shape), 'seconds_at_20fps': xyz.shape[1] / 20,
                       'purpose': 'Pretrained inference; multiple texts use token concatenation, not a learned transition module.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['preview', 'reconstruct', 'train-smoke', 'generate'])
    parser.add_argument('--source', type=Path, help='Preview/reconstruction .npy, or training sample directory')
    parser.add_argument('--out', type=Path, help='New output directory (must not already exist)')
    parser.add_argument('--vq', type=Path, default=VQ)
    parser.add_argument('--gpt', type=Path, default=GPT)
    parser.add_argument('--text', action='append', help='Repeat for multiple motion segments')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--sample', action='store_true', help='Sample tokens instead of greedy decoding')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--no-gif', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.max_tokens, args.threads) < 1:
        parser.error('--steps, --max-tokens and --threads must be positive')
    from datetime import datetime
    out = args.out or ROOT / 'output/local_runs' / (args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    print(f'Output: {out.resolve()}', flush=True)
    if args.mode == 'preview':
        source = args.source or ROOT / 'motion.npy'
        xyz = np.load(source, allow_pickle=False)
        render(xyz, out / 'motion.gif', source.name)
        write_report(out, {'mode': args.mode, 'source': str(source), 'shape': list(xyz.shape)})
    else:
        {'reconstruct': reconstruct, 'train-smoke': train_smoke, 'generate': generate}[args.mode](args, out)


if __name__ == '__main__':
    main()
