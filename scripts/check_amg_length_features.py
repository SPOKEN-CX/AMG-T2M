#!/usr/bin/env python3
"""Check byte alignment, batch independence, no label leakage, and cache identity."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch_geometric.data import Batch
from clip.simple_tokenizer import SimpleTokenizer
import spacy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_amg_length_features import align_words,normalize,sha256
from dataset.dataset_amg_lengths import CaptionLengthDataset
from models.MotionLenEstimator.cached_length import CachedMotionLengthEstimator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--spacy-model',required=True)
    args=p.parse_args()
    manifest=json.loads((args.features/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['smoke_limit']
    for name,digest in manifest['cache_sha256'].items():
        assert sha256(args.features/name)==digest,f'Cache checksum changed: {name}'
    values=np.load(args.features/'features.npy',mmap_mode='r')
    for start in range(0,len(values),10000):
        assert np.isfinite(values[start:start+10000]).all()
    nlp=spacy.load(args.spacy_model,disable=['ner','lemmatizer'])
    tokenizer=SimpleTokenizer()
    texts=["A person doesn't stop walking.","Someone's left-hand turns — then stops.",
           "Café walkers jump!",'Jump.','a person '+('walks forward '*100)]
    for text in texts:
        doc=nlp(normalize(text))
        positions,edges,truncated=align_words(doc,tokenizer)
        assert all(1<=i<=75 for pos in positions for i in pos)
        assert all(0<=a<len(positions) and 0<=b<len(positions) for a,b in edges)
        if not truncated:
            # Every real BPE position must belong to at least one word.
            assert set(i for pos in positions for i in pos)==set(range(1,len(tokenizer.encode(doc.text))+1))
    train=CaptionLengthDataset(args.features,'train')
    val=CaptionLengthDataset(args.features,'val')
    test=CaptionLengthDataset(args.features,'test')
    # Regression: dataset IDs are not graph node indices and must not be offset.
    selected=[test[i] for i in (0, 512, len(test)-1)]
    batched=Batch.from_data_list(selected)
    assert batched.record_id.tolist()==[0,512,len(test)-1], 'Dataset IDs changed during PyG batching'
    ids=[{r['id'] for r in dataset.rows} for dataset in (train,val,test)]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    torch.manual_seed(123)
    model=CachedMotionLengthEstimator().eval().cuda()
    a=train[0]
    b=max((train[i] for i in range(32)),key=lambda g:g.num_nodes)
    with torch.no_grad():
        single=model(Batch.from_data_list([a]).cuda())
        mixed=model(Batch.from_data_list([a,b]).cuda())[:1]
        assert torch.allclose(single,mixed,atol=1e-5,rtol=1e-5),'Prediction depends on batch composition'
        changed=a.clone()
        changed.y+=100
        assert torch.allclose(single,model(Batch.from_data_list([changed]).cuda()),atol=1e-6),'Target leakage'
    report={'passed':True,'byte_alignment_cases':len(texts),'batch_independence':True,
            'target_leakage_check':True,'records':manifest['records'],'cache_hashes_verified':True}
    (args.features/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
