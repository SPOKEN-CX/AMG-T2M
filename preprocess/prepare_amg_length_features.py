#!/usr/bin/env python3
"""Cache frozen CLIP word features with exact byte-span BPE/word alignment."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import ftfy
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import clip
from clip.simple_tokenizer import SimpleTokenizer, whitespace_clean
import spacy


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return whitespace_clean(html.unescape(html.unescape(ftfy.fix_text(text)))).strip().lower()


def align_words(doc, tokenizer):
    text = doc.text
    spans, ids = [], []
    for match in tokenizer.pat.finditer(text):
        encoded = ''.join(tokenizer.byte_encoder[b] for b in match.group().encode('utf-8'))
        offset = len(text[:match.start()].encode('utf-8'))
        for piece in tokenizer.bpe(encoded).split(' '):
            size = len(piece.removesuffix('</w>'))
            spans.append((offset, offset + size))
            ids.append(tokenizer.encoder[piece])
            offset += size
    assert ids == tokenizer.encode(text), 'BPE alignment differs from CLIP tokenizer'
    word_positions, kept = [], []
    for word in doc:
        begin = len(text[:word.idx].encode('utf-8'))
        end = begin + len(word.text.encode('utf-8'))
        positions = [i + 1 for i, (a, b) in enumerate(spans[:75]) if a < end and b > begin]
        if positions:
            kept.append(word.i)
            word_positions.append(positions)
    if not kept:
        raise ValueError(f'No aligned words: {text!r}')
    mapping = {old: new for new, old in enumerate(kept)}
    edges = [(mapping[t.i], mapping[t.head.i]) for t in doc
             if t.i in mapping and t.head.i in mapping and t.i != t.head.i]
    return word_positions, edges, len(ids) > 75


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, default=ROOT / 'output/amg_prepare/lengths_full_v2')
    p.add_argument('--output', type=Path, default=ROOT / 'output/amg_prepare/length_features_v1')
    p.add_argument('--clip-path', type=Path, default=ROOT / 'ViT-B-32.pt')
    p.add_argument('--spacy-model', default='en_core_web_sm')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128, help='frozen CLIP cache export batch')
    p.add_argument('--limit', type=int, default=0, help='records per split for smoke only')
    args = p.parse_args()
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Completed cache exists; choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    records, source_hashes, motion_ids = {}, {}, {}
    for split in ('train', 'val', 'test'):
        path = args.records / f'{split}.jsonl'
        source_hashes[split] = sha256(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if args.limit:
            rows = rows[:args.limit]
        assert all(r['split'] == split and r['length_frames'] > 0 for r in rows)
        official = set((ROOT / 'dataset/HumanML3D' / f'{split}.txt').read_text().splitlines())
        assert all(r['id'] in official for r in rows)
        records[split] = rows
        motion_ids[split] = {r['id'] for r in rows}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not motion_ids[a] & motion_ids[b], f'Motion split leakage: {a}/{b}'
    captions = list(dict.fromkeys(r['caption'] for rows in records.values() for r in rows))
    caption_index = {text: i for i, text in enumerate(captions)}
    nlp = spacy.load(args.spacy_model, disable=['ner', 'lemmatizer'])
    tokenizer = SimpleTokenizer()
    graphs, total_words, truncated = [], 0, 0
    for i, doc in enumerate(nlp.pipe((normalize(s) for s in captions), batch_size=256)):
        positions, edges, was_truncated = align_words(doc, tokenizer)
        graphs.append({'positions': positions, 'edges': edges, 'offset': total_words,
                       'nodes': len(positions), 'word_count': len(doc)})
        total_words += len(positions)
        truncated += int(was_truncated)
        if (i + 1) % 5000 == 0:
            print(f'Parsed {i+1}/{len(captions)} captions', flush=True)
    np.save(args.output / 'offsets.npy', np.array([g['offset'] for g in graphs] + [total_words]))
    # Half precision storage only; training converts cached features to float32.
    features = np.lib.format.open_memmap(args.output / 'features.npy', mode='w+', dtype=np.float16,
                                         shape=(total_words, 512))
    model, _ = clip.load(str(args.clip_path), device=args.device, jit=False)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for first in range(0, len(captions), args.batch_size):
            texts = [normalize(s) for s in captions[first:first + args.batch_size]]
            tokens = clip.tokenize(texts, truncate=True).to(args.device)
            x = model.token_embedding(tokens).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2).float()
            for j in range(len(texts)):
                graph = graphs[first + j]
                word_features = torch.stack([x[j, pos].mean(0) for pos in graph['positions']])
                if not torch.isfinite(word_features).all():
                    raise ValueError('Nonfinite frozen CLIP features')
                start = graph['offset']
                features[start:start + graph['nodes']] = word_features.cpu().numpy().astype(np.float16)
            if first == 0 or (first // args.batch_size + 1) % 20 == 0:
                print(f'Encoded {min(first+args.batch_size,len(captions))}/{len(captions)} captions', flush=True)
    features.flush()
    del features
    for graph in graphs:
        del graph['positions']
    (args.output / 'graphs.json').write_text(json.dumps(graphs))
    for split, rows in records.items():
        for row in rows:
            row['feature_index'] = caption_index[row['caption']]
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    # Scale is chosen using training labels alone; labels are not clipped at 196.
    scale = 2 ** math.ceil(math.log2(max(r['length_frames'] for r in records['train'])))
    manifest = {'format': 1, 'complete': True, 'smoke_limit': args.limit,
                'records': {s: len(v) for s, v in records.items()}, 'unique_captions': len(captions),
                'total_words': total_words, 'truncated_captions': truncated, 'length_scale': scale,
                'source_sha256': source_hashes, 'clip_sha256': sha256(args.clip_path),
                'spacy_version': spacy.__version__, 'spacy_model': nlp.meta,
                'feature_definition': 'Frozen CLIP transformer hidden states before ln_final, mean BPE states per spaCy word; byte-span alignment; FP16 storage',
                'graph_definition': 'Directed child-to-head dependency edges; GAT adds self-loops',
                'fps': 20, 'supervision': 'Unclipped full/timed caption segment length in frames'}
    manifest['cache_sha256'] = {name: sha256(args.output/name) for name in
                                ('features.npy','graphs.json','offsets.npy','train.jsonl','val.jsonl','test.jsonl')}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ('records','unique_captions','length_scale','truncated_captions')},indent=2),flush=True)


if __name__ == '__main__':
    main()
