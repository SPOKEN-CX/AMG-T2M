#!/usr/bin/env python3
"""Render genuine model outputs from the explicitly new supervised semantic branch."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox,HPacker,TextArea
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from common import PROJECT,REPO,read_rows,save_json,digest
from analyze import run_tsne,latent_metrics
from render_polished import PALETTE

OUT=PROJECT/'artifacts/semantic_metric_head_v1'
SEED=20261030


def coordinates(folder,key,x,rows,p=30):
    path=folder/f'{key}_tsne_p{p}_s42.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sample_ids'],np.array([r['sample_id'] for r in rows]));return z['coordinates'].copy()
    reduced=PCA(n_components=min(50,len(x)-1,x.shape[1]),svd_solver='full').fit_transform(x)
    xy,kl=run_tsne(reduced,{'tsne':{'iterations':3000}},p,42)
    np.savez_compressed(path,coordinates=xy,sample_ids=np.array([r['sample_id'] for r in rows]),labels=np.array([r['label'] for r in rows]),kl_divergence=kl,perplexity=p,seed=42)
    return xy


def metric(fig,center,m,dimensions):
    parts=[('Silhouette ',8.3,'#26313D',False),(f'{m["macro_silhouette"]:.4f}',9.3,'#147C85',True),
           (' | ',8.,'#9AA3AD',False),('3-NN purity ',8.3,'#26313D',False),(f'{m["macro_neighbor_purity"]:.2%}',9.3,'#147C85',True)]
    pack=HPacker(children=[TextArea(t,textprops={'fontfamily':'STIXGeneral','fontsize':s,'color':c,'weight':'bold' if b else 'normal'}) for t,s,c,b in parts],align='center',pad=0,sep=3)
    fig.add_artist(AnnotationBbox(pack,(center,.102),xycoords=fig.transFigure,box_alignment=(.5,.5),frameon=False,pad=0))


def panel(ax,xy,labels,order):
    center=(xy.min(0)+xy.max(0))/2;span=np.ptp(xy,axis=0)
    ax.set_xlim(center[0]-.555*span[0],center[0]+.555*span[0]);ax.set_ylim(center[1]-.555*span[1],center[1]+.555*span[1]);ax.set_aspect('equal',adjustable='datalim')
    ax.set(xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.add_patch(FancyBboxPatch((0,0),1,1,transform=ax.transAxes,boxstyle='round,pad=0,rounding_size=.007',facecolor='none',edgecolor='#C4CCD4',linewidth=.55,linestyle=(0,(3.4,2.8)),clip_on=False,zorder=0))
    colors=np.array([PALETTE[c] for c in labels]);points=ax.scatter(xy[order,0],xy[order,1],c=colors[order],s=20,alpha=.93,edgecolors='white',linewidths=.15,zorder=3)
    np.testing.assert_array_equal(np.asarray(points.get_offsets()),xy[order])


def pair(folder,rows,keys,titles,metrics,scope,name,representation_description=None,landscape_alignment=False,hide_scope=False):
    fig=plt.figure(figsize=(8.4,3.05),facecolor='white');labels=np.array([r['label'] for r in rows]);order=np.random.default_rng(42).permutation(len(rows));classes=list(dict.fromkeys(labels));coords={}
    dimensions={};rotations={}
    for left,key,title in zip([.022,.440],keys,titles):
        with np.load(folder/f'{key}_features.npz',allow_pickle=False) as z:x=z['features'].copy()
        dimensions[key]=int(x.shape[1])
        if folder.name=='previous_exploratory420' and key=='baseline_raw':
            source=PROJECT/'artifacts/reference_style_action_search_v3/selected_profile/baseline_pca_p30_e12_s42.npz'
            with np.load(source,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['sample_ids'],np.array([r['sample_id'] for r in rows]));xy=z['coordinates'].copy()
        else:xy=coordinates(folder,key,x,rows)
        if landscape_alignment:
            center=xy.mean(0);_,_,vt=np.linalg.svd(xy-center,full_matrices=False)
            rotated=(xy-center)@vt.T
            np.testing.assert_allclose(np.linalg.norm(xy[1:]-xy[:-1],axis=1),np.linalg.norm(rotated[1:]-rotated[:-1],axis=1),rtol=2e-6,atol=2e-5)
            rotations[key]={'orthogonal_rotation':vt.T.tolist(),'center':center.tolist(),'distance_preserving':True}
            xy=rotated
        coords[key]=xy;ax=fig.add_axes((left,.205,.400,.670));panel(ax,xy,labels,order)
        fig.text(left+.2,.951,title,ha='center',va='center',fontsize=10 if 'head' in title else 11,fontweight='bold')
        metric(fig,left+.2,metrics[key],x.shape[1])
    leg=fig.add_axes((.858,.205,.122,.670));leg.set_axis_off()
    leg.add_patch(FancyBboxPatch((0,0),1,1,transform=leg.transAxes,boxstyle='round,pad=0,rounding_size=.024',edgecolor='#D2D8DE',facecolor='#FCFDFE',linewidth=.6,clip_on=False))
    leg.text(.5,.937,'Action',fontsize=9.4,fontweight='bold',va='center',ha='center');leg.plot([.11,.89],[.885,.885],color='#E0E5E9',lw=.55)
    for c,y in zip(classes,np.linspace(.808,.103,len(classes))):
        leg.scatter([.2],[y],s=24,c=PALETTE[c],edgecolors='white',linewidths=.3);leg.text(.37,y,c,fontsize=9.1,va='center')
    leg.set(xlim=(0,1),ylim=(0,1))
    if not hide_scope:
        fig.text(.5,.025,scope,ha='center',va='center',fontsize=6.3,color='#657180')
    for ext in ['pdf','svg']:fig.savefig(folder/f'{name}.{ext}')
    fig.savefig(folder/f'{name}.png',dpi=900);fig.savefig(folder/f'{name}_preview.png',dpi=300);plt.close(fig)
    with (folder/f'{name}_plotted_coordinates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','label','left_x','left_y','right_x','right_y'])
        for i,r in enumerate(rows):w.writerow([r['sample_id'],r['label'],*coords[keys[0]][i],*coords[keys[1]][i]])
    caption=(f'This figure is part of a NEW caption-supervised semantic-head study. Same{len(rows)} motion crops in both panels. '
        f'Left:{titles[0]}; right:{titles[1]}. Raw features are512-dimensional L2-normalized temporal means of nearest codebook embeddings. '
        +(representation_description or 'Semantic features, when displayed, are128-dimensional continuous outputs of a learned MLP fitted only on539 training motion features with caption-derived action labels. ')
        +f'The displayed feature dimensions are{dimensions}. '
        +
        'Action labels are not inputs during motion feature inference or PCA/t-SNE. Identical unsupervised PCA50 and Euclidean t-SNE settings are used: perplexity30, early exaggeration12, seed42,3000 iterations. '
        'Metrics are macro-averaged in the original representation specified above, never computed on or inferred from edited image coordinates. '
        'The head checkpoint is selected on135 development clips; the fixed primary training seed is20261030. All three training seeds and the matched ordinary-VQ head control are reported separately. '
        'Labels await human review. The initial legacy VQ has a history of test-based checkpoint selection, so the new test cohort is not an untouched benchmark for that VQ. '
        'The auxiliary head does not change raw VQ latents or reconstruction. No inherited FID.0301 claim is made for the new branch. '+scope)
    (folder/f'{name}_CAPTION.txt').write_text(caption+'\n',encoding='utf-8')
    save_json(folder/f'{name}_manifest.json',{'same_samples':True,'samples':len(rows),'titles':titles,'head_primary_seed':SEED,
        'all_points_actual_model_predictions':True,'manual_point_movement':False,'synthetic_points':False,'supervised_projection_at_plotting':False,
        'new_supervised_semantic_branch':any('semantic' in k or 'fusion' in k for k in keys),'raw_bottleneck_unchanged':True,'matched_head_training_control':keys[0].startswith(('baseline_semantic','baseline_fusion')),
        'isolates_tmr_causal_effect':False,
        'feature_dimensions':dimensions,
        'scope_displayed_in_figure':not hide_scope,
        'global_landscape_alignment':rotations,
        'metrics':{k:metrics[k] for k in keys},'scope':scope,'ratio':[2520,915],
        'output_sha256':{p.name:digest(p) for p in folder.glob(f'{name}*') if p.is_file() and not p.name.endswith('_manifest.json')}})


def sensitivity(folder,rows,metrics):
    fig,axs=plt.subplots(3,2,figsize=(10,10));labels=np.array([r['label'] for r in rows]);order=np.random.default_rng(42).permutation(len(rows))
    keys=['baseline_raw',f'ours_semantic_s{SEED}']
    for row,p in enumerate([20,30,50]):
        for col,k in enumerate(keys):
            with np.load(folder/f'{k}_features.npz',allow_pickle=False) as z:x=z['features'].copy()
            xy=coordinates(folder,k,x,rows,p);panel(axs[row,col],xy,labels,order)
            axs[row,col].set_title(('Baseline raw VQ' if col==0 else 'Ours + caption-supervised head')+f' | P={p}',fontsize=10)
    fig.suptitle('All preset perplexities | same motion clips | new semantic-head experiment',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(folder/'sensitivity_real.png',dpi=250);fig.savefig(folder/'sensitivity_real.pdf');plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'text.color':'#26313D','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    ev=json.loads((OUT/'evaluation.json').read_text());annotations=json.loads((REPO/'models/Retrival/datasets/annotations/humanml3d/annotations.json').read_text())
    def rec(r):return annotations[r['motion_id']]['path'].removeprefix('M/')
    fit_sources={rec(r) for r in read_rows(OUT/'fit.csv')}
    for name,s in ev.items():
        folder=OUT/name;rows=read_rows(folder/'samples.csv');m=s['metrics'];ours=f'ours_semantic_s{SEED}';baseline=f'baseline_semantic_s{SEED}'
        if name=='previous_exploratory420':
            n=sum(rec(r) in fit_sources for r in rows)
            scope=f'Exploratory420 clips; {n} recordings used to fit the new head. Caption-derived labels await review.'
        else:scope='102 official-test clips; recording-disjoint from official train/val. Caption-derived labels; unequal class counts.'
        pair(folder,rows,['baseline_raw',ours],['(a) Baseline VQ-VAE','(b) Ours + caption-supervised head'],m,scope,'real_semantic_comparison')
        pair(folder,rows,[baseline,ours],['(a) Baseline + same semantic head','(b) Ours + same semantic head'],m,scope+' Matched supervision control.','matched_supervision_control')
        pair(folder,rows,['baseline_raw','ours_raw'],['(a) Baseline VQ-VAE','(b) Original Ours (raw VQ)'],m,scope+' Original-weight reference.','original_vq_reference')
        if name=='previous_exploratory420':sensitivity(folder,rows,m)
    lines=['# 新增语义分支的真实结果','','所有散点来自真实运动经过已有VQ和新训练语义分支的连续输出，未合成、未按类别移动。',
        '原始VQ编码器/码本/解码器冻结；这是新增caption-supervised语义表示，并非原始512维量化潜空间变得分离。旧FID0.0301不能作为新增方法的评价结果。',
        '用539条拟合、135条开发留出选择模型。7类由前置协议固定；不再按模型效果删动作或删个别样本。',
        '102条官方测试候选均与官方train/val原始录制来源不交叉，但旧版VQ有历史测试选择记录，不能称完全未接触的benchmark。动作弱标签尚未人工核验。',
        '', '|数据|表示|Silhouette|3NN纯度|跨split5NN宏准确率|','|---|---|---:|---:|---:|']
    for name,s in ev.items():
        for key in ['baseline_raw','ours_raw',f'baseline_semantic_s{SEED}',f'ours_semantic_s{SEED}']:
            q=s['metrics'][key];lines.append(f'|{name}|{key}|{q["macro_silhouette"]:.4f}|{q["macro_neighbor_purity"]:.2%}|{q["cross_split_5nn"]["macro_accuracy"]:.2%}|')
    lines+=['','同样监督训练的普通VQ分支也必须报告；若两者改善接近，不能将新增监督带来的收益归因于TMR。',
            '420图包含分支训练数据，只适合展示训练后表示结构。102图及跨split检索指标用于检查泛化；少数类别仅3–5段，需扩大人工核验测试集再作期刊结论。',
            '完整卷积VQ微调试跑因与现有GPT训练共享GPU、每步约24秒而停止，未保存更新后的权重；其step0文件不代表微调成果。',
            '三训练种子固定为20261030/31/32，主图固定20261030，未按测试集或图形美观选择种子。',
            '', '重新运行：先semantic_metric_head.py train，再evaluate，最后render_semantic_experiment.py。原始数据、权重、旧图未覆盖。','']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    print('RENDERED',str(OUT),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
