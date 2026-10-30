"""Verify a completed 100-step AMG smoke run and reload its checkpoints."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vqvae import HumanVQVAE


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert (run / 'exit_code.txt').read_text().strip() == '0', 'Training did not exit successfully'
    state = torch.load(run / 'training_last.pth', map_location='cpu')
    assert state['iter'] == 100 and state['scheduler']['last_epoch'] == 100
    assert state['optimizer']['state'] and finite_tree(state), 'Invalid model/optimizer state'
    opt = argparse.Namespace(**state['args'])
    model = HumanVQVAE(opt, opt.nb_code, opt.code_dim, opt.output_emb_width, opt.down_t,
                      opt.stride_t, opt.width, opt.depth, opt.dilation_growth_rate, opt.vq_act, opt.vq_norm).cpu().eval()
    manifest = ROOT / opt.amg_pairs_dir / 'train_pairs.jsonl'
    row = json.loads(manifest.read_text().splitlines()[0])
    motion = np.load(ROOT / 'dataset/HumanML3D/new_joint_vecs' / (row['id'] + '.npy'))[:64]
    meta = ROOT / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    x = torch.from_numpy(((motion - np.load(meta / 'mean.npy')) / np.load(meta / 'std.npy')).astype('float32')).unsqueeze(0)
    checked = []
    for name in ['net_last.pth', 'net_best_fid.pth', 'training_last.pth']:
        ckpt = torch.load(run / name, map_location='cpu')
        model.load_state_dict(ckpt['net'], strict=True)
        assert finite_tree(ckpt['net'])
        with torch.inference_mode():
            output = model(x)
        assert output[0].shape == x.shape and finite_tree(output), name
        checked.append(name)
    periodic = ROOT / 'checkpoints_model' / opt.exp_name
    first = torch.load(periodic / 'model_50.pth', map_location='cpu')
    last = torch.load(periodic / 'model_100.pth', map_location='cpu')
    assert first['iter'] == 50 and last['iter'] == 100
    assert finite_tree(first) and finite_tree(last)
    assert all(torch.equal(v, last['net'][k]) for k, v in state['net'].items())
    encoder_changed = any(not torch.equal(v, last['net'][k]) for k, v in first['net'].items() if 'encoder' in k)
    assert encoder_changed, 'Encoder parameters did not change between iterations 50 and 100'
    events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
    metrics = {}
    for tag in ['./Test/FID', './Train/TotalLoss', './Train/GradNorm', './Train/ContrastiveLoss']:
        values = events.Scalars(tag)
        assert values and all(np.isfinite(v.value) for v in values), tag
        metrics[tag] = [{'step': v.step, 'value': v.value} for v in values]
    assert [v['step'] for v in metrics['./Test/FID']] == [0, 50, 100]
    report = {'passed': True, 'main_iterations': 100, 'warmup_iterations': opt.warm_up_iter,
              'reloaded_checkpoints': checked, 'periodic_iterations': [50, 100],
              'encoder_updated': encoder_changed, 'reconstruction_shape': list(x.shape), 'metrics': metrics}
    (run / 'smoke_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
