#!/usr/bin/env python3
"""Predict raw frame lengths for new captions using a trained length checkpoint."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import json
from pathlib import Path
import sys
import torch
import spacy
from torch_geometric.data import Data,Batch
from clip.simple_tokenizer import SimpleTokenizer
import clip

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from preprocess.prepare_amg_length_features import align_words,normalize,sha256
from models.motion_length_estimator.cached_length import CachedMotionLengthEstimator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--spacy-model',type=str,required=True)
    p.add_argument('--text',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu')
    manifest_path=Path(state['config']['features'])/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert sha256(manifest_path)==state['manifest_sha256']
    assert sha256(ROOT/'ViT-B-32.pt')==manifest['clip_sha256']
    model=CachedMotionLengthEstimator().cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    encoder,_=clip.load(str(ROOT/'ViT-B-32.pt'),device='cuda',jit=False)
    encoder.eval().requires_grad_(False)
    nlp=spacy.load(args.spacy_model,disable=['ner','lemmatizer'])
    texts=[normalize(text) for text in args.text]
    tokenizer=SimpleTokenizer()
    graphs=[]
    with torch.inference_mode():
        tokens=clip.tokenize(texts,truncate=True).cuda()
        x=encoder.token_embedding(tokens).type(encoder.dtype)+encoder.positional_embedding.type(encoder.dtype)
        x=encoder.transformer(x.permute(1,0,2)).permute(1,0,2).float()
        for i,doc in enumerate(nlp.pipe(texts)):
            positions,edges,_=align_words(doc,tokenizer)
            word_features=torch.stack([x[i,pos].mean(0) for pos in positions]).half().float()
            graphs.append(Data(x=word_features,edge_index=torch.tensor(edges,dtype=torch.long,device='cuda').reshape(-1,2).t().contiguous(),
                               word_count=torch.tensor([len(doc)],dtype=torch.float32,device='cuda')))
        lengths=(model(Batch.from_data_list(graphs))*state['config']['length_scale']).cpu().tolist()
    result=[{'caption':text,'pred_frames':length,'pred_seconds':length/20,
             'rounded_frames':max(1,round(length))} for text,length in zip(args.text,lengths)]
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
