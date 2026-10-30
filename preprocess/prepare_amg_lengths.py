#!/usr/bin/env python3
"""Create auditable HumanML3D caption/length records, without fitting a model."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/HumanML3D'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'output/amg_prepare/lengths')
    p.add_argument('--limit', type=int, default=0, help='limit IDs per split for smoke test')
    args = p.parse_args()
    if args.limit < 0:
        p.error('--limit must be non-negative')
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {s: (DATA / (s + '.txt')).read_text().splitlines() for s in ('train','val','test')}
    for i, a in enumerate(splits):
        for b in list(splits)[i + 1:]:
            if set(splits[a]) & set(splits[b]):
                raise ValueError(f'Split overlap: {a} / {b}')
    report = {}
    for split, ids in splits.items():
        if args.limit:
            ids = ids[:args.limit]
        counts = Counter()
        errors = []
        with (args.output / f'{split}.jsonl').open('w', encoding='utf-8') as out:
            for name in ids:
                try:
                    motion = np.load(DATA / 'new_joint_vecs' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                    if motion.ndim != 2 or motion.shape[1] != 263 or not np.isfinite(motion).all():
                        raise ValueError('invalid or nonfinite motion')
                    for line_number, line in enumerate((DATA / 'texts' / (name + '.txt')).read_text(encoding='utf-8').splitlines(), 1):
                        try:
                            parts = line.split('#')
                            if len(parts) < 4 or not parts[0].strip():
                                raise ValueError(f'invalid caption row {line_number}')
                            start, end = float(parts[2]), float(parts[3])
                            start = 0.0 if math.isnan(start) else start
                            end = 0.0 if math.isnan(end) else end
                            if not math.isfinite(start) or not math.isfinite(end):
                                raise ValueError(f'invalid caption timestamps row {line_number}')
                            is_full = start == end == 0
                            first = 0 if is_full else max(0, int(start * 20))
                            last = len(motion) if is_full else min(len(motion), int(end * 20))
                            if last <= first:
                                raise ValueError(f'empty caption segment row {line_number}')
                            row = {'split': split, 'id': name, 'line': line_number, 'caption': parts[0].strip(),
                                   'start_frame': first, 'end_frame': last, 'length_frames': last-first,
                                   'fps': 20, 'caption_kind': 'full' if is_full else 'timed'}
                            out.write(json.dumps(row, ensure_ascii=False) + '\n')
                            counts[row['caption_kind']] += 1
                        except ValueError as exc:
                            errors.append({'id': name, 'line': line_number, 'error': str(exc)})
                except (ValueError, OSError, UnicodeError) as exc:
                    errors.append({'id':name,'error':str(exc)})
        report[split] = {'ids_scanned':len(ids), 'records':sum(counts.values()), 'kinds':dict(counts), 'errors':errors}
        print(split, 'ids', len(ids), 'records', sum(counts.values()), 'errors', len(errors), flush=True)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if any(v['errors'] for v in report.values()):
        print('Some records failed; see audit.json. Do not use the output without reviewing them.')


if __name__ == '__main__':
    main()
