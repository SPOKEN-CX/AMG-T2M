"""Read-only checks of AMG results, TMR source/weight identity and exported features."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TMR = ROOT / 'src/models/retrieval'
sys.path.insert(0, str(TMR))
from src.config import read_config
from src.load import load_model_from_cfg
from src.data.collate import collate_x_dict
from src.data.motion import Normalizer


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cosine(a, b):
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    out = ROOT / 'output/amg_audit/20261028'
    out.mkdir(parents=True, exist_ok=True)
    report = {'runs': {}, 'resources': []}
    for name in ['VQ_base', 'VQ_amg_full']:
        log = (ROOT / 'output/hml3d' / name / 'run.log').read_text()
        config = json.loads(log[log.index('{'):log.index('\n}') + 2])
        matches = [(int(a), float(b)) for a, b in re.findall(r'Eva\. Iter (\d+) :, FID\. ([\d.]+)', log)]
        report['runs'][name] = {'config': config, 'best_validation': min(matches, key=lambda x: x[1]),
                                'last_validation': matches[-1], 'validation_count': len(matches),
                                'best_at_5000_cadence': min((x for x in matches if x[0] % 5000 == 0), key=lambda x: x[1])}
    model_dir = TMR / 'src/models/tmr_humanml3d_guoh3dfeats'
    paths = [ROOT / 'preprocess/prepare_amg_tmr.py', ROOT / 'preprocess/prepare_amg_missing_text.py',
             ROOT / 'src/datasets/dataset_amg_vq.py', TMR / 'encode_motion_1.py', TMR / 'encode_text_1.py',
             TMR / 'src/load.py', TMR / 'src/model/tmr.py', TMR / 'src/model/temos.py',
             TMR / 'src/model/actor.py', TMR / 'src/model/text_encoder.py', model_dir / 'config.json']
    paths += list((model_dir / 'last_weights').glob('*.pt'))
    paths += list((TMR / 'stats/humanml3d/guoh3dfeats').glob('*.pt'))
    paths += list((ROOT / 'output/amg_prepare/models/distilbert-base-uncased').glob('*'))
    for path in paths:
        if path.is_file():
            report['resources'].append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    model = load_model_from_cfg(read_config(str(model_dir)), ckpt_name='last', device='cpu', eval_mode=True)
    report['loaded_modules'] = {}
    for name in ('motion_encoder', 'text_encoder', 'motion_decoder'):
        expected = torch.load(model_dir / 'last_weights' / (name + '.pt'), map_location='cpu')
        actual = getattr(model, name).state_dict()
        matches = actual.keys() == expected.keys() and all(torch.equal(actual[k], v) for k, v in expected.items())
        if not matches:
            raise RuntimeError(f'Weight load mismatch: {name}')
        report['loaded_modules'][name] = {'matches_weight_file': matches,
                                        'parameter_count': sum(p.numel() for p in getattr(model, name).parameters())}
    prepared = ROOT / 'output/amg_prepare/humanml3d_train_ready'
    rows = [json.loads(line) for line in (prepared / 'train_pairs.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(42)
    sample = [rows[i] for i in rng.choice(len(rows), 64, replace=False)]
    motions = np.stack([np.load(prepared / 'motion' / (r['id'] + '.npy')) for r in sample])
    texts = np.stack([np.load(prepared / 'text' / (hashlib.sha256(r['captions'][0].encode()).hexdigest() + '.npy')) for r in sample])
    sim = (texts / np.linalg.norm(texts, axis=-1, keepdims=True)) @ (motions / np.linalg.norm(motions, axis=-1, keepdims=True)).T
    report['pair_diagnostic_not_official_metric'] = {'pairs': 64, 'matched_mean_cosine': float(sim.diagonal().mean()),
        'mismatched_mean_cosine': float(sim[~np.eye(64, dtype=bool)].mean())}
    normalizer = Normalizer(str(TMR / 'stats/humanml3d/guoh3dfeats'))
    report['reencoded_motion_checks'] = []
    with torch.inference_mode():
        for row in [r for r in sample if r['length'] <= 196][:8]:
            motion = torch.from_numpy(np.load(ROOT / 'dataset/HumanML3D/new_joint_vecs' / (row['id'] + '.npy')).astype('float32'))
            actual = model.encode(collate_x_dict([{'x': normalizer(motion), 'length': len(motion)}]), sample_mean=True).numpy()[0]
            old = np.load(prepared / 'motion' / (row['id'] + '.npy'))
            report['reencoded_motion_checks'].append({'id': row['id'], 'max_abs_difference': float(np.abs(old - actual).max()), 'cosine': float(cosine(old, actual))})
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('resources', 'runs')}, indent=2))
    print('Resource inventory and results:', out / 'audit.json')


if __name__ == '__main__':
    main()
