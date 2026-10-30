#!/usr/bin/env python3
"""Single-GPU graph length training; select on val MAE, test the best model once."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset_amg_lengths import CaptionLengthDataset
from models.motion_length_estimator.cached_length import CachedMotionLengthEstimator

ROOT = Path(__file__).resolve().parents[1]


def atomic_save(value, path):
    tmp = str(path) + '.tmp'
    torch.save(value, tmp)
    os.replace(tmp, path)


def metrics(pred, target):
    delta = np.asarray(pred) - np.asarray(target)
    return {'samples': len(delta), 'mae_frames': float(np.abs(delta).mean()),
            'mae_seconds': float(np.abs(delta).mean()/20),
            'rmse_frames': float(np.sqrt((delta**2).mean())),
            'median_absolute_error_frames': float(np.median(np.abs(delta))),
            'within_10_frames': float((np.abs(delta) <= 10).mean())}


@torch.inference_mode()
def evaluate(model, loader, scale, device):
    model.eval()
    pred, target, indices = [], [], []
    for graph in loader:
        graph = graph.to(device)
        output = model(graph)*scale
        if not torch.isfinite(output).all():
            raise RuntimeError('Nonfinite length prediction')
        pred.extend(output.cpu().tolist())
        target.extend(graph.y.cpu().tolist())
        indices.extend(graph.record_id.cpu().tolist())
    if indices != list(range(len(loader.dataset))):
        raise RuntimeError('Evaluation record IDs changed or were reordered during batching')
    return metrics(pred, target), (pred, target, indices)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=ROOT/'output/amg_prepare/length_features_v1')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--total-iter', type=int, default=50000)
    p.add_argument('--eval-iter', type=int, default=1000)
    p.add_argument('--save-iter', type=int, default=1000)
    p.add_argument('--print-iter', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--resume', type=Path)
    p.add_argument('--skip-test', action='store_true', help='smoke runs only; omit final test')
    p.add_argument('--allow-smoke-cache', action='store_true')
    args = p.parse_args()
    if min(args.batch_size,args.total_iter,args.eval_iter,args.save_iter,args.print_iter) < 1:
        p.error('batch size and intervals must be positive')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir/'run.log').exists() and not args.resume:
        raise FileExistsError('Experiment exists; use --resume or a new output')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.out_dir/'run.log')])
    log = logging.getLogger('length')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    manifest_path = args.features/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not manifest['complete'] or (manifest['smoke_limit'] and not args.allow_smoke_cache):
        raise ValueError('Formal training requires a completed full feature cache')
    scale = manifest['length_scale']
    train = CaptionLengthDataset(args.features,'train')
    val = CaptionLengthDataset(args.features,'val')
    if len(train) < args.batch_size:
        raise ValueError('Not enough train records for one batch')
    assert all(0 < r['length_frames'] < scale for r in train.rows)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train,batch_size=args.batch_size,shuffle=True,drop_last=True,
                        num_workers=args.num_workers,pin_memory=True,generator=generator,
                        persistent_workers=args.num_workers>0)
    val_loader = DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    baseline = float(np.median([r['length_frames'] for r in train.rows]))
    baseline_val = metrics([baseline]*len(val),[r['length_frames'] for r in val.rows])
    model = CachedMotionLengthEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,min_lr=1e-6)
    step, best, best_step = 0, float('inf'), 0
    if args.resume:
        state = torch.load(args.resume,map_location='cpu')
        if state['manifest_sha256'] != manifest_hash or state['config']['batch_size'] != args.batch_size:
            raise ValueError('Resume cache identity or batch size changed')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step,best,best_step = state['iter'],state['best_val_mae'],state['best_iter']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng'])
        log.info('Resumed step %d; DataLoader order restarts',step)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(length_scale=scale,architecture='4 graph residual blocks, 4 GAT heads, CLIP width512, dropout0.3',
                  objective='mean L1 on normalized lengths; MAE reported in original frames',
                  manifest_sha256=manifest_hash,train_median_frames=baseline)
    (args.out_dir/'config.json').write_text(json.dumps(config,indent=2))
    (args.out_dir/'baseline_val.json').write_text(json.dumps(baseline_val,indent=2))
    log.info('CONFIG %s',json.dumps(config))
    log.info('Train %d val %d; train-median baseline val MAE %.4f frames',len(train),len(val),baseline_val['mae_frames'])
    writer = SummaryWriter(str(args.out_dir))
    iterator = iter(loader)
    loss_sum, loss_count, started, first_gradient = 0.,0,time.monotonic(),None

    def save():
        atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'iter':step,'best_val_mae':best,'best_iter':best_step,
                     'config':config,'manifest_sha256':manifest_hash,'python_rng':random.getstate(),
                     'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()},
                    args.out_dir/'training_last.pth')
        log.info('Saved full training state at iteration %d',step)

    while step < args.total_iter:
        try:
            graph = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            graph = next(iterator)
        assert graph.num_graphs == args.batch_size
        graph = graph.to(device)
        model.train()
        prediction = model(graph)
        loss = torch.nn.functional.l1_loss(prediction,graph.y/scale)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
        if first_gradient is None:
            first_gradient = float(grad_norm)
            if first_gradient <= 0:
                raise RuntimeError('Zero initial gradient')
            log.info('First batch: graphs=%d L1_frames=%.5f gradient_norm=%.5f',graph.num_graphs,float(loss)*scale,first_gradient)
        optimizer.step()
        step += 1
        loss_sum += float(loss)*scale
        loss_count += 1
        writer.add_scalar('train/mae_frames',float(loss)*scale,step)
        if step % args.print_iter == 0:
            log.info('Train. Iter %d : MAE %.5f frames, lr %.3g, elapsed %.1fs',step,loss_sum/loss_count,optimizer.param_groups[0]['lr'],time.monotonic()-started)
            loss_sum,loss_count = 0.,0
        if step % args.eval_iter == 0 or step == args.total_iter:
            result,_ = evaluate(model,val_loader,scale,device)
            scheduler.step(result['mae_frames'])
            log.info('Val. Iter %d : %s',step,json.dumps(result))
            writer.add_scalar('val/mae_frames',result['mae_frames'],step)
            if result['mae_frames'] < best:
                best,best_step = result['mae_frames'],step
                atomic_save({'model':model.state_dict(),'iter':step,'val':result,'config':config,
                             'manifest_sha256':manifest_hash},args.out_dir/'net_best_mae.pth')
        if step == 20 or step % args.save_iter == 0 or step == args.total_iter:
            save()
            writer.flush()
            if step == 20:
                (args.out_dir/'startup_check.json').write_text(json.dumps({'passed':True,'iteration':step,
                    'actual_batch_size':graph.num_graphs,'loss_finite':True,'first_gradient_norm':first_gradient,
                    'fresh_initialization':args.resume is None},indent=2))
    if not args.skip_test:
        state = torch.load(args.out_dir/'net_best_mae.pth',map_location='cpu')
        model.load_state_dict(state['model'],strict=True)
        test = CaptionLengthDataset(args.features,'test')
        test_loader = DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=0)
        result,(pred,target,indices) = evaluate(model,test_loader,scale,device)
        baseline_test = metrics([baseline]*len(test),[r['length_frames'] for r in test.rows])
        result.update(best_iteration=best_step,best_val_mae_frames=best,train_median_baseline=baseline_test,
                      manifest_sha256=manifest_hash,checkpoint='net_best_mae.pth',unit='frames',fps=20)
        (args.out_dir/'test_metrics.json').write_text(json.dumps(result,indent=2))
        with (args.out_dir/'test_predictions.csv').open('w',newline='') as f:
            out = csv.writer(f)
            out.writerow(['id','line','caption','caption_kind','true_frames','pred_frames','absolute_error_frames'])
            for estimate,truth,index in zip(pred,target,indices):
                row = test.rows[index]
                if truth != row['length_frames']:
                    raise RuntimeError('Prediction metadata does not match its true length')
                out.writerow([row['id'],row['line'],row['caption'],row['caption_kind'],truth,estimate,abs(estimate-truth)])
        log.info('TEST %s',json.dumps(result))
    writer.close()
    (args.out_dir/'completion.json').write_text(json.dumps({'passed':True,'iteration':step,'best_iter':best_step,
        'best_val_mae_frames':best,'batch_size':args.batch_size,'test_completed':not args.skip_test},indent=2))


if __name__ == '__main__':
    main()
