#!/usr/bin/env python3
"""Prepare caption-aligned HumanML3D/TMR pairs without changing the source dataset."""
import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/HumanML3D'
TMR = ROOT / 'models/Retrival'
DEFAULT_OUT = ROOT / 'output/amg_prepare'
INDEX = TMR / 'datasets/annotations/humanml3d/token_embeddings/distilbert-base-uncased_index.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0, help='limit train IDs for a smoke check (0 = all)')
    parser.add_argument('--encode', action='store_true', help='encode prepared pairs using local TMR')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resolve-whitespace', action='store_true', help='match missing captions to a unique index entry after whitespace normalization')
    parser.add_argument('--missing-token-dir', type=Path, help='verified replacement token embeddings for captions absent from the TMR cache')
    args = parser.parse_args()
    if args.window_size < 1 or args.limit < 0 or args.batch_size < 1:
        parser.error('window-size and batch-size must be positive; limit must be non-negative')
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    if args.resolve_whitespace:
        normalized = {}
        for text, row_index in index.items():
            key = re.sub(r'\s+', ' ', text).strip()
            normalized.setdefault(key, []).append(row_index)
    ids = (DATA / 'train.txt').read_text().splitlines()
    if args.limit:
        ids = ids[:args.limit]
    counts = Counter()
    records = []
    missing_captions = []
    errors = []
    for name in ids:
        try:
            motion = np.load(DATA / 'new_joint_vecs' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            if motion.ndim != 2 or motion.shape[1] != 263:
                raise ValueError('expected (T,263) motion')
            if len(motion) < args.window_size:
                counts['short_motion'] += 1
                continue
            if not np.isfinite(motion).all():
                counts['nonfinite_motion'] += 1
                errors.append({'id': name, 'error': 'nonfinite motion'})
                continue
            captions = []
            for line in (DATA / 'texts' / (name + '.txt')).read_text(encoding='utf-8').splitlines():
                parts = line.split('#')
                if len(parts) < 4:
                    raise ValueError('malformed caption')
                start, end = float(parts[2]), float(parts[3])
                start = 0.0 if math.isnan(start) else start
                end = 0.0 if math.isnan(end) else end
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError('nonfinite caption timestamp')
                if start != 0 or end != 0:
                    counts['partial_captions_excluded'] += 1
                    continue
                caption = parts[0].strip()
                if not caption:
                    raise ValueError('empty caption')
                if caption not in index:
                    candidates = normalized.get(re.sub(r'\s+', ' ', caption).strip(), []) if args.resolve_whitespace else []
                    if len(candidates) == 1:
                        index[caption] = candidates[0]
                        counts['whitespace_resolved'] += 1
                    else:
                        missing_captions.append({'id': name, 'caption': caption})
                captions.append(caption)
            if not captions:
                counts['no_full_caption'] += 1
                continue
            records.append({'id': name, 'length': len(motion), 'captions': list(dict.fromkeys(captions))})
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append({'id': name, 'error': str(exc)})
            counts['errors'] += 1
    unresolved = list(missing_captions)
    if args.missing_token_dir and missing_captions:
        metadata = json.loads((args.missing_token_dir / 'metadata.json').read_text())
        if metadata.get('reference_count', 0) < 1 or metadata.get('index_sha256') != hashlib.sha256(INDEX.read_bytes()).hexdigest():
            raise ValueError('Missing token features were not verified against the current TMR index')
        unresolved = []
        checked = {}
        for item in missing_captions:
            caption = item['caption']
            if caption not in checked:
                key = hashlib.sha256(caption.encode('utf-8')).hexdigest()
                path = args.missing_token_dir / (key + '.npy')
                if path.is_file():
                    x = np.load(path, mmap_mode='r', allow_pickle=False)
                    checked[caption] = x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == 768 and np.isfinite(x).all()
                else:
                    checked[caption] = False
            if not checked[caption]:
                unresolved.append(item)
    manifest = args.output / 'train_pairs.jsonl'
    with manifest.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    report = {'train_ids_scanned': len(ids), 'motions': len(records),
              'caption_pairs': sum(len(r['captions']) for r in records),
              'missing_precomputed_text': len(missing_captions), 'counts': dict(counts),
              'errors': errors, 'missing_captions': missing_captions,
              'missing_unresolved_text': len(unresolved), 'missing_unresolved_captions': unresolved,
              'missing_token_dir': str(args.missing_token_dir.resolve()) if args.missing_token_dir else None,
              'window_size': args.window_size,
              'resolve_whitespace': args.resolve_whitespace,
              'policy': 'full-motion captions only; motion embedding is full sequence, VQ input is a random window'}
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Audit:', {k: report[k] for k in ('train_ids_scanned', 'motions', 'caption_pairs', 'missing_precomputed_text', 'counts')}, flush=True)
    if args.encode:
        if errors or unresolved or not records:
            raise RuntimeError('Cannot encode: inspect audit.json for invalid inputs or missing text embeddings')
        encode(records, index, args)


def encode(records, index, args):
    import numpy as np
    import torch
    sys.path.insert(0, str(TMR))
    from src.config import read_config
    from src.load import load_model_from_cfg
    from src.data.motion import Normalizer
    from src.data.collate import collate_x_dict
    model_dir = TMR / 'models/tmr_humanml3d_guoh3dfeats'
    cfg = read_config(str(model_dir))
    model = load_model_from_cfg(cfg, ckpt_name='last', device=args.device, eval_mode=True)
    normalizer = Normalizer(str(TMR / 'stats/humanml3d/guoh3dfeats'))
    token_dir = TMR / 'datasets/annotations/humanml3d/token_embeddings'
    tokens = np.load(token_dir / 'distilbert-base-uncased.npy', mmap_mode='r', allow_pickle=False)
    slices = np.load(token_dir / 'distilbert-base-uncased_slice.npy', mmap_mode='r', allow_pickle=False)
    motions_dir, text_dir = args.output / 'motion', args.output / 'text'
    motions_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    unique_texts = {caption for row in records for caption in row['captions']}
    print(f'Encoding {len(records)} motions and {len(unique_texts)} unique captions on {args.device}', flush=True)
    with torch.inference_mode():
        for offset in range(0, len(records), args.batch_size):
            rows = records[offset:offset + args.batch_size]
            xs = []
            for row in rows:
                motion = torch.from_numpy(np.load(DATA / 'new_joint_vecs' / (row['id'] + '.npy')).astype('float32'))
                motion = normalizer(motion).to(args.device)
                xs.append({'x': motion, 'length': len(motion)})
            latent = model.encode(collate_x_dict(xs), sample_mean=True).cpu().numpy()
            if latent.shape != (len(rows), 256) or not np.isfinite(latent).all():
                raise ValueError('Invalid motion embeddings')
            for row, value in zip(rows, latent):
                np.save(motions_dir / (row['id'] + '.npy'), value)
            if (offset // args.batch_size) % 50 == 0:
                print(f'motions: {min(offset + len(rows), len(records))}/{len(records)}', flush=True)
        texts = sorted(unique_texts)
        for offset in range(0, len(texts), args.batch_size):
            batch = texts[offset:offset + args.batch_size]
            xs = []
            for caption in batch:
                if caption in index:
                    begin, end = slices[index[caption]]
                    values = np.array(tokens[begin:end], dtype='float32', copy=True)
                else:
                    key = hashlib.sha256(caption.encode('utf-8')).hexdigest()
                    values = np.load(args.missing_token_dir / (key + '.npy'), allow_pickle=False).astype('float32')
                x = torch.from_numpy(values).to(args.device)
                xs.append({'x': x, 'length': len(x)})
            latent = model.encode(collate_x_dict(xs), sample_mean=True).cpu().numpy()
            if latent.shape != (len(batch), 256) or not np.isfinite(latent).all():
                raise ValueError('Invalid text embeddings')
            for caption, value in zip(batch, latent):
                np.save(text_dir / (hashlib.sha256(caption.encode('utf-8')).hexdigest() + '.npy'), value)
            if (offset // args.batch_size) % 50 == 0:
                print(f'texts: {min(offset + len(batch), len(texts))}/{len(texts)}', flush=True)
    print('Encoding completed:', args.output, flush=True)


if __name__ == '__main__':
    main()
