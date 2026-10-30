"""Verify the completed AMG VQ run and freeze its best validation checkpoint."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vqvae import HumanVQVAE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'output/hml3d/VQ_amg_full')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/hml3d/VQ_amg_frozen')
    args = parser.parse_args()
    source = args.source_dir.resolve()
    target = args.output.resolve()
    if (source / 'status.txt').read_text().strip() not in ('completed', 'completed and verified') or (source / 'exit_code.txt').read_text().strip() != '0':
        raise RuntimeError('AMG VQ full training did not complete successfully')
    state = torch.load(source / 'training_last.pth', map_location='cpu')
    if state['iter'] != state['args']['total_iter'] or state['scheduler']['last_epoch'] != state['iter']:
        raise RuntimeError('Unexpected final AMG VQ iteration')
    best_iteration, best_fid = int(state['best'][1]), float(state['best'][0])
    best = torch.load(source / 'net_best_fid.pth', map_location='cpu')['net']
    periodic = torch.load(ROOT / 'checkpoints_model' / state['args']['exp_name'] / f'model_{best_iteration}.pth', map_location='cpu')['net']
    if best.keys() != periodic.keys() or not all(torch.equal(value, periodic[key]) for key, value in best.items()):
        raise RuntimeError('Best VQ weights do not match the recorded best iteration')
    if not all(torch.isfinite(value).all() for value in best.values()):
        raise RuntimeError('Nonfinite VQ weights')
    opt = argparse.Namespace(**state['args'])
    net = HumanVQVAE(opt, opt.nb_code, opt.code_dim, opt.output_emb_width, opt.down_t,
                    opt.stride_t, opt.width, opt.depth, opt.dilation_growth_rate, opt.vq_act, opt.vq_norm).cuda().eval()
    net.load_state_dict(best, strict=True)
    row = json.loads((ROOT / opt.amg_pairs_dir / 'train_pairs.jsonl').read_text().splitlines()[0])
    motion = np.load(ROOT / 'dataset/HumanML3D/new_joint_vecs' / (row['id'] + '.npy'))[:64]
    meta = ROOT / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    x = torch.from_numpy(((motion - np.load(meta / 'mean.npy')) / np.load(meta / 'std.npy')).astype('float32')).unsqueeze(0).cuda()
    with torch.inference_mode():
        tokens = net.encode(x)
        reconstruction = net.forward_decoder(tokens)
    if tokens.shape != (1, 16) or tokens.min() < 0 or tokens.max() >= 512 or reconstruction.shape != x.shape or not torch.isfinite(reconstruction).all():
        raise RuntimeError('VQ token encode/decode smoke failed')
    digest = sha256(source / 'net_best_fid.pth')
    target.mkdir(parents=True, exist_ok=True)
    frozen = target / 'net_best_fid.pth'
    if frozen.exists():
        if sha256(frozen) != digest:
            raise RuntimeError('Existing frozen VQ differs; refusing to overwrite it')
    else:
        temporary = target / 'net_best_fid.pth.tmp'
        shutil.copyfile(source / 'net_best_fid.pth', temporary)
        if sha256(temporary) != digest:
            raise RuntimeError('VQ copy checksum mismatch')
        temporary.replace(frozen)
    report = {'verified': True, 'source': str(source / 'net_best_fid.pth'), 'frozen': str(frozen),
              'vq_sha256': digest, 'completed_iteration': state['iter'], 'best_iteration': best_iteration,
              'best_validation_fid': best_fid, 'encode_shape': list(tokens.shape),
              'decode_shape': list(reconstruction.shape), 'bf16_supported': torch.cuda.is_bf16_supported()}
    (target / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
