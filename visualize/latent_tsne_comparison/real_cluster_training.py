#!/usr/bin/env python3
"""Actual caption-supervised VQ fine-tuning, with saved protocols and matched controls.
This produces NEW model weights; it is not a rendering change or the old FID=.0301 model.
"""
import argparse, copy, json, math, time
from argparse import Namespace
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import PROJECT, REPO, digest, read_rows, write_rows, save_json
from features import motion_batch, normalize
from analyze import latent_metrics
import sys
sys.path.insert(0, str(REPO))
from models.vqvae import HumanVQVAE

CLASSES=['run','jump','sit','wave','kick','dance','throw']
SPECS={'ours':REPO/'checkpoints/self_train/VQVAE/t2m/net_best_fid.pth',
       'baseline':REPO/'output/hml3d/VQ_base/net_last.pth'}
SEED=20261030

def supcon(z, labels, tau=.1):
    z=F.normalize(z,dim=-1)
    logits=z@z.T/tau
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive=(labels[:,None]==labels[None,:]) & ~eye
    if (positive.sum(1)==0).any(): raise ValueError('Every training anchor needs a positive')
    logp=logits-torch.logsumexp(logits.masked_fill(eye,float('-inf')),dim=1,keepdim=True)
    return -(logp.masked_fill(~positive,0).sum(1)/positive.sum(1)).mean()

def reconstruction(pred,x,commit):
    rec=F.smooth_l1_loss(pred,x)
    local=F.smooth_l1_loss(pred[...,4:67],x[...,4:67])
    return 4*rec+.5*local+.02*commit,rec

def load_net(path,device):
    model=HumanVQVAE(Namespace(dataname='t2m',quantizer='ema_reset',mu=.99),512,512,512,2,2,512,3,3,'relu',None).to(device)
    ck=torch.load(path,map_location='cpu')
    model.load_state_dict(ck['net'],strict=True)
    # A net-only legacy checkpoint has no EMA accumulators. Preserve its learned
    # codebook rather than silently reinitializing it on the first training batch.
    q=model.vqvae.quantizer
    q.init=True
    q.code_count=torch.full((q.nb_code,),2.,device=device)
    q.code_sum=q.codebook.detach().clone()*2
    model.projection_layer.requires_grad_(False) # unused, never treat as trained features
    return model

def macro_score(x,rows):
    labels=np.array([r['label'] for r in rows]); src=np.array([r['source_id'] for r in rows])
    m=latent_metrics(normalize(x),labels,src,3)
    pcs={c:{'samples':int((labels==c).sum()),
        'silhouette':float(m['silhouette_per_sample'][labels==c].mean()),
        'neighbor_purity':float(m['knn_purity_per_sample'][labels==c].mean())} for c in CLASSES}
    return {'macro_silhouette':float(np.mean([v['silhouette'] for v in pcs.values()])),
            'macro_neighbor_purity':float(np.mean([v['neighbor_purity'] for v in pcs.values()])),
            'per_class':pcs}

@torch.no_grad()
def evaluate(model,x,rows):
    model.eval(); features=[]; pre=[]; rec_sum=0.; ppl_sum=0.
    for start in range(0,len(x),16):
        bx=x[start:start+16]
        pred,commit,ppl,post=model(bx)
        features.append(post.cpu().numpy())
        pre.append(model.vqvae.encoder(bx.permute(0,2,1)).mean(-1).cpu().numpy())
        rec_sum+=float(F.smooth_l1_loss(pred,bx))*len(bx)
        ppl_sum+=float(ppl)*len(bx)
    raw=np.concatenate(features); metric=macro_score(raw,rows)
    metric.update(reconstruction_smooth_l1=rec_sum/len(x),mean_batch_code_perplexity=ppl_sum/len(x))
    return metric,normalize(raw),normalize(np.concatenate(pre))

def prepare(out):
    census=out/'census.json'
    if not census.exists(): raise FileNotFoundError('Run the caption/source census first')
    source=read_rows(out/'train_candidates.csv'); train=[]; dev=[]
    for c in CLASSES:
        group=[r for r in source if r['label']==c]
        group.sort(key=lambda r:__import__('hashlib').sha256(f'{SEED}:{r["source_id"]}'.encode()).hexdigest())
        n=max(4,round(len(group)*.2));dev+=group[:n];train+=group[n:]
    write_rows(out/'fit.csv',train);write_rows(out/'dev.csv',dev)
    plan={'kind':'new caption-supervised contrastive VQ fine-tuning; not the unchanged legacy model',
        'classes_fixed_before_scoring':CLASSES,'seed':SEED,'steps_per_arm':600,'eval_every':100,
        'initial_checkpoints':{k:{'path':str(p),'sha256':digest(p)} for k,p in SPECS.items()},
        'fit_counts':dict(Counter(r['label'] for r in train)),
        'development_counts':dict(Counter(r['label'] for r in dev)),
        'development_note':'Stratified source-disjoint holdout of official train for NEW fine-tuning only; the initial pretrained VQs have seen official train',
        'test_note':'102 caption-clean official test clips, excluding recordings in any official train or val; historical legacy checkpoint was previously selected using a historical test protocol',
        'class_supervision':'Weak caption-derived action labels used ONLY for fitting; never supplied for feature inference or t-SNE',
        'optimizer':'AdamW, encoder and decoder lr=2e-5, betas=(.9,.99), no weight decay; EMA mu=.99',
        'batch':'8 distinct clips per each of the fixed 7 actions; 56 total; identical batches in both arms',
        'objective':'4*SmoothL1(all263)+.5*SmoothL1(local_xyz)+.02*commit + lambda*supervised_contrastive(post_mean, tau=.1)',
        'lambda_rule':'geometric mean across arms of reconstruction encoder gradient norm / SupCon encoder gradient norm on a single identical training batch; clamp [.1,50]; fixed thereafter',
        'selection':'maximum development original512D macro cosine silhouette among checkpoints with reconstruction <=1.15*initial and code perplexity >=.5*initial; tie break 3NN purity; step0 included',
        'test_selection':False,'data_filter_uses_model_metrics':False,'labels_human_reviewed':False,
        'inference':'motion only -> encoder -> nearest codebook embeddings -> temporal mean -> L2; no classifier logits, no hard class assignments, no label-dependent transformations',
        'projection':'PCA50 + unsupervised t-SNE, P30/E12/seed42/3000iters, identical settings in both panels; no coordinate edits',
        'files_sha256':{p.name:digest(p) for p in [out/'fit.csv',out/'dev.csv',out/'test_candidates.csv']}}
    save_json(out/'training_protocol_before_scores.json',plan)
    return train,dev,plan

def train_all(out):
    if (out/'training_protocol_before_scores.json').exists(): raise FileExistsError('Preserve completed/partial run; use a new output directory')
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    if not torch.cuda.is_available(): raise RuntimeError('This bounded full-VQ training pilot requires CUDA')
    device='cuda';torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    fit,dev,plan=prepare(out)
    cfg=json.loads((PROJECT/'config.json').read_text())
    xfit=torch.from_numpy(motion_batch(fit,cfg)).to(device)
    xdev=torch.from_numpy(motion_batch(dev,cfg)).to(device)
    yfit=torch.tensor([CLASSES.index(r['label']) for r in fit],device=device)
    groups=[np.flatnonzero(np.array([r['label'] for r in fit])==c) for c in CLASSES]
    rng=np.random.default_rng(SEED);batches=[np.concatenate([rng.choice(g,8,replace=False) for g in groups]) for _ in range(600)]
    calibration={}
    for arm,path in SPECS.items():
        model=load_net(path,device).eval();idx=batches[0]
        pred,commit,ppl,z=model(xfit[idx]);rl,rec=reconstruction(pred,xfit[idx],commit);cl=supcon(z,yfit[idx])
        params=tuple(model.vqvae.encoder.parameters())
        nr=torch.stack([v.norm() for v in torch.autograd.grad(rl,params,retain_graph=True)]).norm()
        nc=torch.stack([v.norm() for v in torch.autograd.grad(cl,params)]).norm()
        calibration[arm]={'reconstruction_grad_norm':float(nr),'supcon_grad_norm':float(nc),'unclamped_lambda':float(nr/nc)}
        del model,pred,commit,ppl,z,rl,rec,cl,nr,nc;torch.cuda.empty_cache()
    weight=float(np.clip(np.exp(np.mean([np.log(v['unclamped_lambda']) for v in calibration.values()])),.1,50))
    save_json(out/'gradient_calibration.json',{'arms':calibration,'shared_lambda':weight})
    print('CALIBRATION',json.dumps({'arms':calibration,'shared_lambda':weight}),flush=True)
    all_history={};selected={}
    for arm,path in SPECS.items():
        torch.manual_seed(SEED);model=load_net(path,device)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,betas=(.9,.99),weight_decay=0)
        initial,feats,pre=evaluate(model,xdev,dev)
        best=initial;best_step=0;history=[{'step':0,**initial,'reconstruction_ratio':1.,'eligible':True}]
        torch.save({'net':model.state_dict(),'iteration':0,'experiment':plan,'metrics':initial},out/f'{arm}_best.pth')
        np.savez_compressed(out/f'{arm}_initial_dev_features.npz',post_mean=feats,pre_mean=pre,labels=np.array([r['label'] for r in dev]),sample_ids=np.array([r['sample_id'] for r in dev]))
        print(arm,'INITIAL',json.dumps(initial),flush=True)
        started=time.monotonic();running=[]
        for step,idx in enumerate(batches,1):
            model.train();pred,commit,ppl,z=model(xfit[idx])
            rloss,rec=reconstruction(pred,xfit[idx],commit);closs=supcon(z,yfit[idx]);loss=rloss+weight*closs
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
            running.append((float(rec),float(closs),float(loss)))
            if step%50==0:
                avg=np.mean(running,axis=0);running=[]
                print(f'{arm} step={step}/600 rec={avg[0]:.5f} supcon={avg[1]:.5f} total={avg[2]:.5f} grad={grad:.3f} elapsed={time.monotonic()-started:.1f}s',flush=True)
            if step%100==0:
                metric,feats,pre=evaluate(model,xdev,dev)
                ratio=metric['reconstruction_smooth_l1']/initial['reconstruction_smooth_l1']
                eligible=ratio<=1.15 and metric['mean_batch_code_perplexity']>=.5*initial['mean_batch_code_perplexity']
                history.append({'step':step,**metric,'reconstruction_ratio':ratio,'eligible':eligible})
                if eligible and (metric['macro_silhouette'],metric['macro_neighbor_purity'])>(best['macro_silhouette'],best['macro_neighbor_purity']):
                    best=metric;best_step=step
                    torch.save({'net':model.state_dict(),'iteration':step,'experiment':plan,'metrics':metric,'lambda_supcon':weight},out/f'{arm}_best.pth')
                    np.savez_compressed(out/f'{arm}_best_dev_features.npz',post_mean=feats,pre_mean=pre,labels=np.array([r['label'] for r in dev]),sample_ids=np.array([r['sample_id'] for r in dev]))
                all_history[arm]=history;save_json(out/'development_history.json',all_history)
                print(arm,'DEV',json.dumps({'step':step,'silhouette':metric['macro_silhouette'],'3nn':metric['macro_neighbor_purity'],'reconstruction_ratio':ratio,'eligible':eligible,'best_step':best_step}),flush=True)
        q=model.vqvae.quantizer
        torch.save({'net':model.state_dict(),'iteration':600,'experiment':plan,'optimizer':opt.state_dict(),'ema_sum':q.code_sum,'ema_count':q.code_count,'lambda_supcon':weight},out/f'{arm}_last_training.pth')
        selected[arm]={'path':str(out/f'{arm}_best.pth'),'sha256':digest(out/f'{arm}_best.pth'),'step':best_step,'metrics':best,'initial_metrics':initial}
        save_json(out/'selected_models.json',selected)
        del model,opt,pred,commit,ppl,z,loss,rloss,closs;torch.cuda.empty_cache()
    print('TRAINING COMPLETE',json.dumps({k:{'step':v['step'],'silhouette':v['metrics']['macro_silhouette'],'purity':v['metrics']['macro_neighbor_purity']} for k,v in selected.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=PROJECT/'artifacts/real_cluster_training_v1');args=p.parse_args()
    train_all(args.output.resolve())
if __name__=='__main__': main()
