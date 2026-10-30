#!/usr/bin/env python3
"""Recover missing DistilBERT token features after verifying the existing cache."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'src/models/retrieval/datasets/annotations/humanml3d/token_embeddings'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--reference-count', type=int, default=64)
    args = p.parse_args()
    if args.batch_size < 1 or args.reference_count < 1:
        p.error('batch-size and reference-count must be positive')
    model = AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    index = json.loads((CACHE / 'distilbert-base-uncased_index.json').read_text())
    slices = np.load(CACHE / 'distilbert-base-uncased_slice.npy', mmap_mode='r')
    stored = np.load(CACHE / 'distilbert-base-uncased.npy', mmap_mode='r')

    def features(texts):
        inputs = tokenizer(texts, padding=True, return_tensors='pt')
        with torch.inference_mode():
            values = model(**inputs).last_hidden_state.cpu().numpy()
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        return [v[:n].copy() for v, n in zip(values, lengths)]

    # Spread reference samples across the existing index rather than checking only one caption.
    texts = list(index)
    positions = np.linspace(0, len(texts) - 1, min(args.reference_count, len(texts)), dtype=int)
    reference = [texts[i] for i in positions]
    max_error = 0.0
    max_mean_error = 0.0
    min_cosine = 1.0
    for start in range(0, len(reference), args.batch_size):
        batch = reference[start:start + args.batch_size]
        for caption, actual in zip(batch, features(batch)):
            begin, end = slices[index[caption]]
            expected = np.asarray(stored[begin:end])
            if expected.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise RuntimeError(f'Invalid reference token features for {caption!r}')
            diff = np.abs(actual - expected)
            mean_error = float(diff.mean())
            largest_error = float(diff.max())
            denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
            if np.any(denominator == 0):
                raise RuntimeError(f'Zero-length reference token vector for {caption!r}')
            cosine = float(np.min(np.sum(actual * expected, axis=1) / denominator))
            if mean_error > 1e-3 or largest_error > 0.05 or cosine < 0.9999:
                raise RuntimeError(f'DistilBERT output differs from cache for {caption!r}: mean={mean_error:.6g}, max={largest_error:.6g}, min cosine={cosine:.8f}')
            max_error = max(max_error, largest_error)
            max_mean_error = max(max_mean_error, mean_error)
            min_cosine = min(min_cosine, cosine)
    captions = sorted({row['caption'] for row in json.loads(args.audit.read_text())['missing_captions']})
    args.output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(captions), args.batch_size):
        batch = captions[start:start + args.batch_size]
        for caption, value in zip(batch, features(batch)):
            if value.ndim != 2 or value.shape[1] != 768 or not np.isfinite(value).all():
                raise ValueError(f'Invalid token embedding for {caption!r}')
            name = hashlib.sha256(caption.encode('utf-8')).hexdigest()
            np.save(args.output / (name + '.npy'), value)
        print(f'missing captions: {min(start + len(batch), len(captions))}/{len(captions)}', flush=True)
    metadata = {'reference_count': len(reference), 'reference_max_absolute_error': max_error,
                'reference_max_mean_absolute_error': max_mean_error, 'reference_min_token_cosine': min_cosine,
                'reference_thresholds': {'mean_absolute_error': 1e-3, 'max_absolute_error': 0.05, 'min_token_cosine': 0.9999}, 'missing_unique_captions': len(captions),
                'model_dir': str(args.model_dir.resolve()),
                'model_sha256': hashlib.sha256((args.model_dir / 'model.safetensors').read_bytes()).hexdigest(),
                'index_sha256': hashlib.sha256((CACHE / 'distilbert-base-uncased_index.json').read_bytes()).hexdigest()}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
