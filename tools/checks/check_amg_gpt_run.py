"""Check a completed AMG GPT run, its frozen VQ identity, tokens and checkpoints."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preprocess.prepare_amg_gpt import sha256
from models.t2m_trans import Text2Motion_Transformer
from datasets.dataset_tokenize import VQMotionDataset


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--expected-iter', type=int, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    if (run / 'exit_code.txt').read_text().strip() != '0':
        raise RuntimeError('Training did not exit successfully')
    state = torch.load(run / 'training_last.pth', map_location='cpu')
    opt = argparse.Namespace(**state['args'])
    if state['iter'] != args.expected_iter or state['scheduler']['last_epoch'] != args.expected_iter or not finite_tree(state):
        raise RuntimeError('Invalid training checkpoint')
    frozen = (ROOT / opt.resume_pth).resolve().parent / 'manifest.json'
    manifest = json.loads(frozen.read_text())
    identity = json.loads((run / 'vq_identity.json').read_text())
    if identity['vq_sha256'] != manifest['vq_sha256'] or sha256(ROOT / opt.resume_pth) != identity['vq_sha256']:
        raise RuntimeError('VQ identity changed')
    model = Text2Motion_Transformer(opt.nb_code, opt.embed_dim_gpt, opt.clip_dim, opt.block_size,
                                   opt.num_layers, opt.n_head_gpt, opt.drop_out_rate, opt.ff_rate).eval()
    for name in ('training_last.pth', 'net_last.pth', 'net_best_fid.pth'):
        weights = state['trans'] if name == 'training_last.pth' else torch.load(run / name, map_location='cpu')['trans']
        model.load_state_dict(weights, strict=True)
        if not finite_tree(weights):
            raise RuntimeError(f'Nonfinite weights: {name}')
        with torch.inference_mode():
            output = model(torch.zeros((1, 2), dtype=torch.long), torch.zeros((1, opt.clip_dim)))
        if not torch.isfinite(output).all():
            raise RuntimeError(f'Invalid forward pass: {name}')
    tokens = ROOT / 'dataset/HumanML3D' / opt.vq_name
    files = list(tokens.glob('*.npy'))
    expected = set(VQMotionDataset('t2m', unit_length=2 ** opt.down_t).name_list)
    actual = {file.stem for file in files}
    if actual != expected:
        raise RuntimeError(f'Token ID mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}')
    for file in files:
        value = np.load(file, allow_pickle=False)
        if value.ndim != 2 or value.shape[0] != 1 or not 1 <= value.shape[1] <= opt.block_size - 2 or not np.issubdtype(value.dtype, np.integer) or value.min() < 0 or value.max() >= opt.nb_code:
            raise RuntimeError(f'Invalid VQ tokens: {file}')
    events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
    loss = events.Scalars('./Loss/train')
    fids = events.Scalars('./Test/FID')
    if not loss or not fids or not all(np.isfinite(v.value) for v in loss + fids):
        raise RuntimeError('Training loss or validation metrics missing/invalid')
    report = {'passed': True, 'iteration': state['iter'], 'vq_sha256': identity['vq_sha256'],
              'vq_name': opt.vq_name, 'microbatch_size': opt.batch_size,
              'grad_accum_steps': getattr(opt, 'grad_accum_steps', 1),
              'effective_batch_size': opt.batch_size * getattr(opt, 'grad_accum_steps', 1),
              'validation_max_batches': getattr(opt, 'validation_max_batches', 0),
              'tokens_checked': len(files), 'last_printed_loss': loss[-1].value,
              'validation_fid': [{'step': v.step, 'value': v.value} for v in fids]}
    (run / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
