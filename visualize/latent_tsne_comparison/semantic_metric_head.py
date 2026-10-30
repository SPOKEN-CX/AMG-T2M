#!/usr/bin/env python3
"""New learned motion-only semantic heads, with validation selection and matched controls.
All shown points are genuine predictions. This is an added caption-supervised branch,
not an edited t-SNE and not evidence that the unchanged VQ bottleneck has improved.
"""
import argparse, json, sys, math, time, csv
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import PROJECT, REPO, digest, read_rows, write_rows, save_json
from features import FrozenEncoder, motion_batch, normalize
from real_cluster_training import CLASSES, SPECS, supcon, macro_score
from threadpoolctl import threadpool_limits

class SemanticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128))
    def forward(self,x): return F.normalize(self.layers(x),dim=-1)


def get_features(rows,cfg,out,arm,name):
    path=out/f'{arm}_{name}_vq_features.npz';ids=np.array([r['sample_id'] for r in rows])
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sample_ids'],ids)
            assert str(z['checkpoint_sha256'])==digest(SPECS[arm])
            return z['post_mean'].copy()
    lookup={}
    for cache in [PROJECT/'artifacts/legacy_contrast_exploration_v1'/('pool_tmr_legacy.npz' if arm=='ours' else 'pool_baseline_last.npz'),
                  PROJECT/'artifacts/reference_style_action_search_v3/selected_profile'/('tmr_b512_features.npz' if arm=='ours' else 'baseline_features.npz')]:
        with np.load(cache,allow_pickle=False) as z:
            assert str(z['checkpoint_sha256'])==digest(SPECS[arm])
            lookup.update({str(i):x for i,x in zip(z['sample_ids'],z['post_mean'])})
    x=np.zeros((len(rows),512),dtype=np.float32);missing=[]
    for i,r in enumerate(rows):
        if r['sample_id'] in lookup: x[i]=lookup[r['sample_id']]
        else: missing.append(i)
    if missing:
        model=FrozenEncoder(cfg,SPECS[arm]).cpu().eval();raw=motion_batch([rows[i] for i in missing],cfg)
        with torch.inference_mode():
            for start in range(0,len(missing),16):
                x[np.array(missing[start:start+16])]=normalize(model(torch.from_numpy(raw[start:start+16]))[1].numpy())
        del model,raw
    assert np.isfinite(x).all()
    np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=2e-6)
    np.savez_compressed(path,post_mean=x,sample_ids=ids,labels=np.array([r['label'] for r in rows]),checkpoint_sha256=digest(SPECS[arm]))
    print('FEATURES',arm,name,'N',len(rows),'newly_encoded',len(missing),flush=True)
    return x

@torch.no_grad()
def embed(head,x):
    head.eval();return head(torch.from_numpy(x)).numpy()


def train(out,parent):
    if (out/'protocol_before_scores.json').exists(): raise FileExistsError('Use a new output directory; preserve previous selection')
    out.mkdir(parents=True,exist_ok=True)
    fit=read_rows(parent/'fit.csv');dev=read_rows(parent/'dev.csv');cfg=json.loads((PROJECT/'config.json').read_text())
    seeds=[20261030,20261031,20261032]
    protocol={
        'kind':'NEW caption-supervised semantic metric head over genuine frozen VQ features',
        'classes':CLASSES,'architecture':'512 -> Linear256 -> LayerNorm -> GELU -> Dropout(.1) -> Linear128 -> L2',
        'loss':'supervised contrastive, tau=.1, training labels only; no classifier probability or label input at inference',
        'optimizer':'AdamW lr3e-4, weight_decay1e-3, betas(.9,.99)',
        'steps_per_seed_arm':1200,'seeds':seeds,'primary_seed':seeds[0],'batch':'8 samples per fixed action, 56 per batch',
        'model_selection':'maximum macro cosine silhouette on135 development clips; require 3NN purity >= the same arm raw VQ baseline; no test metrics used',
        'feature_at_inference':'motion -> frozen VQ -> actual temporal-mean codebook vectors -> learned continuous128D semantic embedding; no action labels, no hard assignment, no point relocation',
        'control':'train the identical head on ordinary VQ features with identical data, batches and seeds; report all arms and seeds',
        'reconstruction':'frozen VQ encoder/codebook/decoder remain unchanged; head is an auxiliary semantic representation and does not modify reconstruction',
        'scope':'not the unchanged512D raw VQ latent; not attributable solely to TMR; not the historical FID.0301 model result',
        'data_selection':'fixed7 classes; all caption-clean candidates from earlier census; no per-sample score filtering',
        'fitting_counts':dict(Counter(r['label'] for r in fit)),'development_counts':dict(Counter(r['label'] for r in dev)),
        'heldout_test_counts':json.loads((parent/'census.json').read_text())['test']['recording_disjoint_counts'],
        'development_note':'source-disjoint only for the new head; pretrained VQ saw official train; no subject-level independence claimed',
        'test_note':'official test records exclude any original recording in train or val; legacy VQ checkpoint previously selected under a historical test protocol, so this is not an untouched benchmark',
        'label_review':'caption-derived weak labels, not human verified',
        'tsne':'unsupervisedPCA50 + P30/E12/seed42/3000iter on each side, same settings and same samples; no image/coordinate editing',
        'hashes':{str(p):digest(p) for p in [parent/'fit.csv',parent/'dev.csv',parent/'test_candidates.csv',*SPECS.values()]}}
    save_json(out/'protocol_before_scores.json',protocol)
    write_rows(out/'fit.csv',fit);write_rows(out/'dev.csv',dev)
    torch.set_num_threads(4);rng0=np.random.default_rng(seeds[0]);labels=np.array([r['label'] for r in fit])
    group=[np.flatnonzero(labels==c) for c in CLASSES];ys=torch.tensor([CLASSES.index(c) for c in labels])
    results={};history={}
    for arm in ['ours','baseline']:
        xf=get_features(fit,cfg,out,arm,'fit');xd=get_features(dev,cfg,out,arm,'dev');rawscore=macro_score(xd,dev)
        results[arm]={'raw_development_metrics':rawscore,'seeds':{}}
        xt=torch.from_numpy(xf)
        for seed in seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);head=SemanticHead()
            opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-3,betas=(.9,.99))
            best=None;beststep=None;hist=[];started=time.monotonic()
            for step in range(1,1201):
                idx=np.concatenate([rng.choice(g,8,replace=False) for g in group]);head.train()
                z=head(xt[idx]);loss=supcon(z,ys[idx]);opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
                if step%100==0:
                    pred=embed(head,xd);m=macro_score(pred,dev)
                    eligible=m['macro_neighbor_purity']>=rawscore['macro_neighbor_purity']
                    hist.append({'step':step,'training_loss':float(loss),**m,'eligible':eligible})
                    if eligible and (best is None or (m['macro_silhouette'],m['macro_neighbor_purity'])>(best['macro_silhouette'],best['macro_neighbor_purity'])):
                        best=m;beststep=step
                        torch.save({'head':head.state_dict(),'architecture':protocol['architecture'],'iteration':step,'seed':seed,
                            'frozen_vq_checkpoint':str(SPECS[arm]),'frozen_vq_sha256':digest(SPECS[arm]),
                            'protocol_sha256':digest(out/'protocol_before_scores.json'),'development_metrics':m},out/f'{arm}_head_s{seed}.pth')
                    print(f'{arm} seed{seed} step{step} loss{float(loss):.4f} devS{m["macro_silhouette"]:.4f} dev3NN{m["macro_neighbor_purity"]:.2%} eligible{eligible} best{beststep} elapsed{time.monotonic()-started:.1f}s',flush=True)
            if best is None: raise RuntimeError(f'No head checkpoint satisfies validation guard for{arm}/{seed}')
            results[arm]['seeds'][str(seed)]={'step':beststep,'metrics':best,'head':str(out/f'{arm}_head_s{seed}.pth'),'sha256':digest(out/f'{arm}_head_s{seed}.pth')}
            history[f'{arm}_{seed}']=hist;save_json(out/'development_history.json',history);save_json(out/'selected_heads.json',results)
    print('HEAD TRAINING COMPLETE',flush=True)


def finalize(out,parent):
    if (out/'evaluation.json').exists(): raise FileExistsError('Final held-out evaluation already exists; preserve it')
    selected=json.loads((out/'selected_heads.json').read_text());protocol=json.loads((out/'protocol_before_scores.json').read_text())
    cfg=json.loads((PROJECT/'config.json').read_text());test=read_rows(parent/'test_candidates.csv')
    prior=read_rows(PROJECT/'artifacts/reference_style_action_search_v3/selected_profile/samples.csv')
    datasets={'source_disjoint_test102':test,'previous_exploratory420':prior};summary={}
    fit=read_rows(out/'fit.csv');fit_labels=np.array([r['label'] for r in fit])
    save_json(out/'additional_metrics_before_test_scores.json',{
        'metric':'5-NN majority-vote classification against the fixed539 training reference bank, cosine distance',
        'summary':'macro accuracy across the fixed7 actions; no test neighbors or labels used to construct the reference bank',
        'used_for_model_selection':False})
    def cross_knn(train_vectors,test_vectors,rows):
        nearest=np.argsort(-(test_vectors@train_vectors.T),axis=1,kind='stable')[:,:5]
        prediction=np.array([max(CLASSES,key=lambda c:(int((fit_labels[ix]==c).sum()),-CLASSES.index(c))) for ix in nearest])
        y=np.array([r['label'] for r in rows])
        per_class={c:float((prediction[y==c]==c).mean()) for c in CLASSES}
        return {'macro_accuracy':float(np.mean(list(per_class.values()))),'micro_accuracy':float((prediction==y).mean()),'per_class':per_class}
    for name,rows in datasets.items():
        target=out/name;target.mkdir(exist_ok=True);write_rows(target/'samples.csv',rows)
        summary[name]={'samples':len(rows),'counts':dict(Counter(r['label'] for r in rows)),'metrics':{}}
        for arm in ['ours','baseline']:
            x=get_features(rows,cfg,out,arm,name);summary[name]['metrics'][f'{arm}_raw']=macro_score(x,rows)
            xfit=get_features(fit,cfg,out,arm,'fit')
            summary[name]['metrics'][f'{arm}_raw']['cross_split_5nn']=cross_knn(xfit,x,rows)
            np.savez_compressed(target/f'{arm}_raw_features.npz',features=x,labels=np.array([r['label'] for r in rows]),sample_ids=np.array([r['sample_id'] for r in rows]))
            for seed in protocol['seeds']:
                ck=torch.load(out/f'{arm}_head_s{seed}.pth',map_location='cpu');head=SemanticHead();head.load_state_dict(ck['head'],strict=True)
                predicted=embed(head,x)
                np.savez_compressed(target/f'{arm}_semantic_s{seed}_features.npz',features=predicted,
                    sample_ids=np.array([r['sample_id'] for r in rows]),labels=np.array([r['label'] for r in rows]),head_sha256=digest(out/f'{arm}_head_s{seed}.pth'))
                summary[name]['metrics'][f'{arm}_semantic_s{seed}']=macro_score(predicted,rows)
                summary[name]['metrics'][f'{arm}_semantic_s{seed}']['cross_split_5nn']=cross_knn(embed(head,xfit),predicted,rows)
                if seed==protocol['primary_seed']:
                    print('EVAL',name,arm,'rawS',summary[name]['metrics'][f'{arm}_raw']['macro_silhouette'],'semanticS',summary[name]['metrics'][f'{arm}_semantic_s{seed}']['macro_silhouette'],'semantic3NN',summary[name]['metrics'][f'{arm}_semantic_s{seed}']['macro_neighbor_purity'],flush=True)
    save_json(out/'evaluation.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--parent',type=Path,default=PROJECT/'artifacts/real_cluster_training_v1');p.add_argument('--output',type=Path,default=PROJECT/'artifacts/semantic_metric_head_v1');a=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if a.stage=='train':train(a.output.resolve(),a.parent.resolve())
        else:finalize(a.output.resolve(),a.parent.resolve())
if __name__=='__main__':main()
