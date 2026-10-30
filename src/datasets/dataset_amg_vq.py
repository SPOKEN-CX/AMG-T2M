"""Caption-aligned HumanML3D samples for the experimental AMG VQ path."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


class AMGPairedVQDataset(Dataset):
    def __init__(self, pairs_dir, window_size=64):
        self.pairs_dir = Path(pairs_dir)
        self.window_size = window_size
        if window_size < 1:
            raise ValueError('window_size must be positive')
        report = json.loads((self.pairs_dir / 'audit.json').read_text(encoding='utf-8'))
        if report['window_size'] != window_size:
            raise ValueError('Prepared window size differs from training window size')
        if report.get('missing_unresolved_text', report['missing_precomputed_text']) or report['errors']:
            raise ValueError('Preparation audit contains errors; inspect audit.json')
        self.records = [json.loads(line) for line in (self.pairs_dir / 'train_pairs.jsonl').read_text(encoding='utf-8').splitlines() if line]
        if not self.records:
            raise ValueError('No prepared training motions')
        self.motion_dir = ROOT / 'dataset/HumanML3D/new_joint_vecs'
        meta = ROOT / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        self.mean = np.load(meta / 'mean.npy')
        self.std = np.load(meta / 'std.npy')
        if self.mean.shape != (263,) or self.std.shape != (263,) or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid VQ normalization statistics')
        for row in self.records:
            if not row['captions'] or row['length'] < window_size:
                raise ValueError(f"Invalid prepared row: {row['id']}")
            files = [self.pairs_dir / 'motion' / (row['id'] + '.npy')]
            files.extend(self.text_feature_path(caption) for caption in row['captions'])
            missing = next((path for path in files if not path.is_file()), None)
            if missing:
                raise FileNotFoundError(f'Prepared feature missing: {missing}')

    def text_feature_path(self, caption):
        key = hashlib.sha256(caption.encode('utf-8')).hexdigest()
        return self.pairs_dir / 'text' / (key + '.npy')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        motion = np.load(self.motion_dir / (row['id'] + '.npy'), allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) != row['length']:
            raise ValueError(f"Motion changed since preparation: {row['id']}")
        start = random.randint(0, len(motion) - self.window_size)
        motion = (motion[start:start + self.window_size] - self.mean) / self.std
        caption = random.choice(row['captions'])
        motion_feature = np.load(self.pairs_dir / 'motion' / (row['id'] + '.npy'), allow_pickle=False)
        text_feature = np.load(self.text_feature_path(caption), allow_pickle=False)
        if motion_feature.shape != (256,) or text_feature.shape != (256,) or not np.isfinite(motion_feature).all() or not np.isfinite(text_feature).all() or not np.isfinite(motion).all():
            raise ValueError(f"Invalid motion/text feature: {row['id']}")
        return motion.astype(np.float32), caption, motion_feature.astype(np.float32), text_feature.astype(np.float32)
