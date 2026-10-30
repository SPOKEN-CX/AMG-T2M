#!/usr/bin/env python3
"""Equal-energy fusion retains motion geometry while using the learned semantic branch.
A single fixed transform is applied to every sample, without inference labels.
"""
import json,csv,argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits
from common import PROJECT,REPO,read_rows,write_rows,save_json,digest
from real_cluster_training import macro_score,CLASSES
from render_semantic_experiment import pair
from semantic_metric_head import SemanticHead,embed

BASE=PROJECT/'artifacts/semantic_metric_head_v1';OUT=BASE/'fusion_equal_v1';SEED=20261030

def main(mode='equal'):
    global OUT
    OUT=BASE/('fusion_equal_v1' if mode=='equal' else 'fusion_soft_v1')
    OUT.mkdir(exist_ok=True)
    calibration={};weight=.5
    if mode=='soft':
        torch.set_num_threads(1)
        for arm in ['baseline','ours']:
            with np.load(BASE/f'{arm}_fit_vq_features.npz',allow_pickle=False) as z:raw=z['post_mean'].copy()
            h=SemanticHead();h.load_state_dict(torch.load(BASE/f'{arm}_head_s{SEED}.pth',map_location='cpu')['head'],strict=True)
            semantic=embed(h,raw);upper=np.triu_indices(len(raw),1)
            dr=np.maximum(0,1-raw@raw.T)[upper];ds=np.maximum(0,1-semantic@semantic.T)[upper]
            calibration[arm]={'training_median_raw_cosine_distance':float(np.median(dr)),'training_median_semantic_cosine_distance':float(np.median(ds))}
        raw_scale=np.sqrt(np.prod([q['training_median_raw_cosine_distance'] for q in calibration.values()]))
        sem_scale=np.sqrt(np.prod([q['training_median_semantic_cosine_distance'] for q in calibration.values()]))
        weight=float(.25*raw_scale/(.25*raw_scale+sem_scale))
    tag='w50' if mode=='equal' else 'soft'
    protocol={'kind':'new motion-semantic fusion representation; original raw VQ is unchanged',
        'rule':f'concat(sqrt({1-weight})*L2(raw512D), sqrt({weight})*L2(learned_semantic128D)) -> unit640D vector',
        'weight_semantic':weight,'primary_weight_selected_before_fusion_scoring':True,
        'training_only_unlabeled_distance_calibration':calibration,
        'soft_weight_rule':'semantic median distance contribution is one quarter of the motion contribution, with shared geometric-mean scales across both arms; no evaluation data or labels used in calibration' if mode=='soft' else None,
        'justification':'retain raw motion variation instead of discarding it in a strongly supervised head; soft mode limits the median semantic distance contribution using training-only scales',
        'cosine_distance_identity':f'D_fusion(i,j)={1-weight}*D_raw(i,j)+{weight}*D_semantic(i,j)',
        'same_fixed_transform_for_all_samples':True,'labels_at_inference':False,'manual_point_movement':False,
        'head_training':'539 fit clips,135 development holdout; caption-supervised SupCon; fixed seed20261030',
        'control':'ordinary VQ receives identical head architecture, supervision and fusion weight',
        'scope':'not an unchanged original raw VQ result; not a strict single-variable TMR ablation; legacy test history prevents untouched-benchmark claims',
        'test_history':'102 source-disjoint test clips have now been evaluated for the strong-head representation; this fusion is a post-hoc exploratory variant, not a fresh confirmatory test',
        'figure':'same input samples, unsupervised PCA50+t-SNEP30/E12/seed42/3000iterations, no class-conditional changes',
        'input_evaluation_sha256':digest(BASE/'evaluation.json')}
    save_json(OUT/'protocol_before_fusion_scores.json',protocol)
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'text.color':'#26313D','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    whole={};evidence={}
    for dataset in ['previous_exploratory420','source_disjoint_test102']:
        source=BASE/dataset;dest=OUT/dataset;dest.mkdir(exist_ok=True);rows=read_rows(source/'samples.csv');write_rows(dest/'samples.csv',rows)
        metrics={};evidence[dataset]={}
        for arm in ['baseline','ours']:
            with np.load(source/f'{arm}_raw_features.npz',allow_pickle=False) as z:raw=z['features'].copy();ids=z['sample_ids'].copy();labels=z['labels'].copy()
            with np.load(source/f'{arm}_semantic_s{SEED}_features.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['sample_ids'],ids);semantic=z['features'].copy()
            fusion=np.concatenate([np.sqrt(1-weight)*raw,np.sqrt(weight)*semantic],axis=1).astype(np.float32)
            np.testing.assert_allclose(np.linalg.norm(fusion,axis=1),1,atol=2e-6)
            dr=np.clip(pairwise_distances(raw,metric='cosine'),0,2);ds=np.clip(pairwise_distances(semantic,metric='cosine'),0,2);df=np.clip(pairwise_distances(fusion,metric='cosine'),0,2)
            # Float32 640-term dot products incur about 1e-6 absolute rounding error.
            np.testing.assert_allclose(df,(1-weight)*dr+weight*ds,atol=2e-6)
            assert (df>=(1-weight)*dr-2e-6).all()
            evidence[dataset][arm]={'max_cosine_identity_error':float(np.abs(df-((1-weight)*dr+weight*ds)).max()),'geometry_check_atol':2e-6,'minimum_preserved_fraction_of_each_raw_cosine_distance':1-weight,'dimensions':640}
            for key,x in [(f'{arm}_raw',raw),(f'{arm}_fusion_{tag}',fusion)]:
                np.savez_compressed(dest/f'{key}_features.npz',features=x,sample_ids=ids,labels=labels)
                metrics[key]=macro_score(x,rows)
        whole[dataset]={'samples':len(rows),'metrics':metrics}
        if dataset=='previous_exploratory420':scope=f'Exploratory420 clips;296 fit recordings. Motion/semantic weights={1-weight:.1%}/{weight:.1%}; caption labels await review.'
        else:scope='102 recording-disjoint test clips; exploratory fusion variant, weak labels and unequal class counts.'
        description=(f'The fused640D representation concatenates512D raw VQ motion features and128D learned semantic-head features, each unit normalized and multiplied by sqrt({1-weight}) and sqrt({weight}) respectively. '
            f'Thus cosine distances use motion/semantic weights{1-weight}/{weight}, retaining at least{1-weight:.2%} of every raw motion cosine distance. '
            'This is one fixed feature transform for every sample; it uses no class label at inference. The semantic head was fitted with weak caption labels. ')
        pair(dest,rows,['baseline_raw',f'ours_fusion_{tag}'],['(a) Baseline VQ-VAE','(b) Ours + motion–semantic fusion'],metrics,scope,'real_fusion_comparison',description,landscape_alignment=mode=='soft')
        pair(dest,rows,[f'baseline_fusion_{tag}',f'ours_fusion_{tag}'],['(a) Baseline + same fusion','(b) Ours + same fusion'],metrics,scope+' Matched-head/fusion control.','matched_fusion_control',description,landscape_alignment=mode=='soft')
        print(dataset,{k:(round(v['macro_silhouette'],4),round(v['macro_neighbor_purity'],4)) for k,v in metrics.items()},flush=True)
    save_json(OUT/'fusion_geometry_checks.json',evidence);save_json(OUT/'evaluation.json',whole)
    lines=['# 保留动作变化的真实融合表示','','这不是P图：所有点来自运动输入、冻结VQ、训练后的连续语义分支及固定等权融合，之后用不接收类别标签的PCA+t-SNE绘制。',
        f'新表示为640维：原始运动512维与学得语义128维的余弦距离权重为{1-weight:.4f}/{weight:.4f}。所有样本、所有类别统一使用同一变换。',
        f'数学与数值检查：融合余弦距离=(1-w)×原始运动余弦距离+w×语义余弦距离；保留每对样本至少{1-weight:.2%}的原始余弦距离。',
        '该方法仍含新增caption-supervised分支，不是原始TMR-VQ潜空间，不继承旧FID.0301，也不能单独证明TMR因果收益。',
        '420图有296个录制来源参与分支拟合；102测试来源与官方train/val不重合，但旧VQ有历史测试选择记录，且该融合权重属于看过强分支结果后的探索性变体。标签未经人工核验。',
        '', '|数据|表示|Silhouette|3NN纯度|','|---|---|---:|---:|']
    for dataset,v in whole.items():
        for k,m in v['metrics'].items():lines.append(f'|{dataset}|{k}|{m["macro_silhouette"]:.4f}|{m["macro_neighbor_purity"]:.2%}|')
    lines+=['','普通VQ也使用相同监督分支和融合权重的对照图在每个数据目录的matched_fusion_control.pdf。','原始VQ编码器、码本、解码器和重建行为都未因本分支改变；这项修改仅用于新的辅助语义表示。','']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['equal','soft'],default='equal');a=p.parse_args()
    with threadpool_limits(limits=1):main(a.mode)
