#!/usr/bin/env python3
"""Transparent whole-action subset of an exact matched original scientific figure.
No sample-wise outcome filtering, no t-SNE refit, no movement of individual clusters.
"""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import PROJECT,digest,read_rows,write_rows,save_json
from analyze import latent_metrics
from render_polished import PALETTE
from render_clean_reference import metric_label
from threadpoolctl import threadpool_limits

SOURCE=PROJECT/'artifacts/legacy_contrast_exploration_v1/recommended_mature_comparison'
OUT=PROJECT/'artifacts/legacy_contrast_exploration_v1/overlap_review_v2'
FAMILIES=['baseline','tmr_b512']
VIEW_SCALE=.88
MARKER=9.


def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve completed outputs; use a new directory')
    OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((SOURCE/'report.json').read_text());audit=json.loads((SOURCE/'selection_audit.json').read_text())
    rows=read_rows(SOURCE/'samples.csv');ids=np.array([r['sample_id'] for r in rows]);labels=np.array([r['label'] for r in rows]);classes=list(report['class_counts'])
    sources=np.array([audit['source_path_by_sample_id'][i] for i in ids]);xy={};features={};diagnostics={}
    inputs=[SOURCE/'report.json',SOURCE/'samples.csv',SOURCE/'selection_audit.json',SOURCE/'polished_clean_v1/comparison_clean.png']
    for family in FAMILIES:
        featpath=SOURCE/f'{family}_features.npz';coordpath=SOURCE/report['journal_design']['coordinate_files'][family];inputs.extend([featpath,coordpath])
        with np.load(featpath,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sample_ids'],ids);np.testing.assert_array_equal(z['labels'],labels)
            assert str(z['checkpoint_sha256'])==report['selected'][family]['sha256'];features[family]=z['post_mean'].copy()
        with np.load(coordpath,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['sample_ids'],ids);np.testing.assert_array_equal(z['labels'],labels);xy[family]=z['coordinates'].copy()
        m=latent_metrics(features[family],labels,sources,3)
        diagnostics[family]={c:{'samples':int((labels==c).sum()),'original512D_3nn_purity':float(m['knn_purity_per_sample'][labels==c].mean())} for c in classes}
    removed=min(classes,key=lambda c:np.mean([diagnostics[f][c]['original512D_3nn_purity'] for f in FAMILIES]))
    assert removed=='throw'
    keep=labels!=removed;assert int(keep.sum())==602
    plan={'operation':'post-hoc whole-action subset with geometry-preserving display compaction','source_study':str(SOURCE),
        'exact_matched_original_png':str(SOURCE/'polished_clean_v1/comparison_clean.png'),
        'source_image_identification':'Uploaded7560x2745 image matched source PNG pixel-for-pixel after RGB conversion',
        'removed_action':removed,'excluded_samples':int((~keep).sum()),'retained_samples':int(keep.sum()),
        'selection_rule':'Remove the single action with the lowest original512D cosine3NN purity averaged equally across BOTH models on the full700 cohort',
        'diagnostics_before_exclusion':diagnostics,'selection_is_posthoc':True,'label_error_not_established':True,'independent_confirmation':False,
        'individual_sample_outcome_filtering':False,'both_panels_keep_exactly_same_sample_ids':True,
        'coordinate_refit':False,'individual_cluster_repositioning':False,'features_changed':False,'new_training':False,
        'view_scale_same_in_both_panels':VIEW_SCALE,'original_marker_size':6.4,'new_marker_size':MARKER,
        'data_split':'train+val','labels_human_reviewed':False,
        'input_sha256':{str(p):digest(p) for p in inputs}}
    save_json(OUT/'selection_and_display_plan.json',plan);write_rows(OUT/'excluded_throw_98.csv',[r for r,k in zip(rows,keep) if not k])
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'text.color':'#26313D','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    summary={}
    for name,mask in [('five_class_602_same_projection',keep),('full_six_700_compact',np.ones(len(rows),bool))]:
        target=OUT/name;target.mkdir(exist_ok=True);rr=[r for r,k in zip(rows,mask) if k];ll=labels[mask];cc=[c for c in classes if c in ll];met={}
        write_rows(target/'samples.csv',rr)
        for family in FAMILIES:
            m=latent_metrics(features[family][mask],ll,sources[mask],3)
            pcs={c:{'samples':int((ll==c).sum()),'silhouette':float(m['silhouette_per_sample'][ll==c].mean()),'neighbor_purity':float(m['knn_purity_per_sample'][ll==c].mean())} for c in cc}
            met[family]={'macro_silhouette':float(np.mean([q['silhouette'] for q in pcs.values()])),
                         'macro_neighbor_purity':float(np.mean([q['neighbor_purity'] for q in pcs.values()])),'per_class':pcs}
            np.savez_compressed(target/f'{family}_unchanged_projection.npz',coordinates=xy[family][mask],features=features[family][mask],sample_ids=ids[mask],labels=ll)
        summary[name]={'samples':len(rr),'class_counts':dict(Counter(ll)),'metrics':met}
        fig=plt.figure(figsize=(8.4,3.05),facecolor='white');original=plt.figure(figsize=(8.4,3.05),facecolor='white')
        order=np.random.default_rng(42).permutation(len(rr));colors=np.array([PALETTE[c] for c in ll]);ratios={}
        for left,family,title in zip([.022,.440],FAMILIES,['(a) Baseline VQ-VAE','(b) Ours (w/ Contrastive Learning)']):
            ax=fig.add_axes((left,.205,.400,.670));ref=original.add_axes((left,.205,.400,.670));full=xy[family];center=(full.min(0)+full.max(0))/2;span=np.ptp(full,axis=0)
            for view,scale in [(ax,VIEW_SCALE),(ref,1.)]:
                pad=.555/scale;view.set_xlim(center[0]-pad*span[0],center[0]+pad*span[0]);view.set_ylim(center[1]-pad*span[1],center[1]+pad*span[1]);view.set_aspect('equal',adjustable='datalim')
                view.set(xticks=[],yticks=[])
                for spine in view.spines.values():spine.set_visible(False)
            ax.add_patch(FancyBboxPatch((0,0),1,1,transform=ax.transAxes,boxstyle='round,pad=0,rounding_size=.007',facecolor='none',edgecolor='#C4CCD4',linewidth=.55,linestyle=(0,(3.4,2.8)),clip_on=False,zorder=0))
            p=full[mask];artist=ax.scatter(p[order,0],p[order,1],c=colors[order],s=MARKER,alpha=.93,edgecolors='white',linewidths=.15,zorder=3)
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()),p[order])
            fig.text(left+.200,.951,title,ha='center',va='center',fontsize=11,fontweight='bold');metric_label(fig,left+.200,met[family])
            # Reproduce the full source plot's data bounds before deriving a
            # common isotropic view change. Empty reference axes would choose
            # a different equal-aspect adjustment in Matplotlib.
            ref.scatter(full[:,0],full[:,1],s=1,alpha=0)
            original.canvas.draw()
            for getter,setter in [(ref.get_xlim,ax.set_xlim),(ref.get_ylim,ax.set_ylim)]:
                lo,hi=getter();middle=(lo+hi)/2;half=(hi-lo)/2/VIEW_SCALE
                setter(middle-half,middle+half)
            ax.set_aspect('equal',adjustable='box')
            fig.canvas.draw()
            new=ax.transData.transform([[0,0],[1,0],[0,1]]);old=ref.transData.transform([[0,0],[1,0],[0,1]])
            rx=np.linalg.norm(new[1]-new[0])/np.linalg.norm(old[1]-old[0]);ry=np.linalg.norm(new[2]-new[0])/np.linalg.norm(old[2]-old[0])
            np.testing.assert_allclose([rx,ry],[VIEW_SCALE,VIEW_SCALE],rtol=1e-6);ratios[family]=[float(rx),float(ry)]
        plt.close(original)
        leg=fig.add_axes((.858,.205,.122,.670));leg.set_axis_off();leg.add_patch(FancyBboxPatch((0,0),1,1,transform=leg.transAxes,boxstyle='round,pad=0,rounding_size=.024',edgecolor='#D2D8DE',facecolor='#FCFDFE',linewidth=.6,clip_on=False))
        leg.text(.5,.937,'Action',fontsize=9.4,fontweight='bold',va='center',ha='center');leg.plot([.11,.89],[.885,.885],color='#E0E5E9',lw=.55)
        for c,y in zip(cc,np.linspace(.808,.103,len(cc))):leg.scatter([.2],[y],s=24,c=PALETTE[c],edgecolors='white',linewidths=.3);leg.text(.37,y,c,fontsize=9.1,va='center')
        leg.set(xlim=(0,1),ylim=(0,1))
        scope='Exploratory action subset: throw excluded from BOTH panels; 602 paired clips.' if name.startswith('five') else 'Full six-action control: all700 paired clips retained.'
        fig.text(.5,.030,scope,fontsize=6.7,color='#657180',ha='center',va='center')
        for ext in ['pdf','svg']:fig.savefig(target/f'comparison_compact.{ext}')
        fig.savefig(target/'comparison_compact.png',dpi=900);fig.savefig(target/'comparison_compact_preview.png',dpi=300);plt.close(fig)
        with (target/'plotted_coordinates.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['sample_id','label','baseline_x','baseline_y','tmr_x','tmr_y'])
            for i,r in zip(np.flatnonzero(mask),rr):w.writerow([r['sample_id'],r['label'],*xy['baseline'][i],*xy['tmr_b512'][i]])
        caption=(f'Geometry-preserving exploratory action-subset display of the original700-clip HumanML3D train+validation comparison. {scope} '
            'The throw class was selected for exclusion because it had the lowest same-class3NN purity averaged over both models in original normalized512D VQ features. '
            'This is post-hoc action-level selection, not label verification or evidence of overall model improvement. No retained sample was filtered by its performance. '
            'For both models, all retained motion IDs, windows, labels,512D features, and2D coordinates are identical to the corresponding originals. '
            'PCA50+t-SNE(P30,E12,seed42,3000iterations) was fitted to the full700 clips before the whole-action subset was displayed; t-SNE was not refitted. '
            f'Original512D cosine silhouette and3NN purity are recalculated on the displayed class subset. Both panels use the same isotropic{VIEW_SCALE:.0%} display scale relative to the full original and marker area{MARKER:g}pt squared. '
            'Neither cluster centers nor within-/between-cluster distance ratios were altered. All points remain in view. '
            'Baseline uses the mature net_last checkpoint; Ours uses the historical net_best_fid checkpoint; no new training or semantic head was used. '
            'Caption-derived labels await human review. The legacy directory reports FID.0301, but evaluation protocol and exact184000-step checkpoint identity cannot be independently verified. This is not a controlled TMR ablation.')
        (target/'FIGURE_CAPTION.txt').write_text(caption+'\n',encoding='utf-8')
        save_json(target/'report.json',summary[name])
        save_json(target/'render_manifest.json',{'operation':plan['operation'],'all_remaining_feature_and_coordinate_values_unchanged':True,
            'display_scale_ratio_verified':ratios,'same_sample_ids_in_both_panels':True,'subset_scope_visible_in_figure':True,'t_sne_refit':False,
            'pointwise_deletion_by_score':False,'class_excluded':removed if name.startswith('five') else None,'metrics_recomputed_on_displayed_subset':True,
            'png_pixels':[7560,2745],'preview_pixels':[2520,915],'point_size':MARKER,
            'source_original':str(SOURCE/'polished_clean_v1/comparison_clean.png'),'outputs_sha256':{p.name:digest(p) for p in target.iterdir() if p.is_file()}})
    save_json(OUT/'summary.json',summary)
    lines=['# 对上传原图的动作子集与紧凑显示修改','','已逐像素定位上传原图到:'+str(SOURCE/'polished_clean_v1/comparison_clean.png'),
        '主图仅从两侧同时排除投掷类别的全部98条；保留602条的标签、窗口、原始特征、二维坐标逐项不变。不按Ours的单个点是否交缠删样本。',
        '投掷两模型原始512维3NN同类比例分别为61.90%和75.85%，两侧平均为六类中最低。混杂不能证明标签错误。',
        f'两侧显示比例同时变为最初700点原图的{VIEW_SCALE:.0%}，点间距统一缩短{1-VIEW_SCALE:.0%}；类内/类间距离的比值保持不变。散点面积由最初6.4变为{MARKER:g}pt²。',
        '外框、两图大小及7560×2745导出比例保持一致。图中显式注明探索性类别子集；保留完整700条相同风格对照。',
        '图下指标在各自显示的原始512维特征上重新计算。排除类别简化了任务，分数上升不代表整体模型能力提升。',
        '', '|版本|普通Silhouette|OursSilhouette|普通3NN|Ours3NN|','|---|---:|---:|---:|---:|']
    for n,v in summary.items():
        b,t=[v['metrics'][f] for f in FAMILIES];lines.append(f'|{n}|{b["macro_silhouette"]:.4f}|{t["macro_silhouette"]:.4f}|{b["macro_neighbor_purity"]:.2%}|{t["macro_neighbor_purity"]:.2%}|')
    lines+=['','本次只修改该旧版六类图；未使用上一轮新增语义分支或新的训练权重。其他实验进度保留，未对其测试集继续评分。','']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(digest(p)==sha for p,sha in plan['input_sha256'].items())
    print(json.dumps({'output':str(OUT),'removed_action':removed,'samples_remaining':int(keep.sum()),'view_scale':VIEW_SCALE,'metrics':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--view-scale',type=float,default=VIEW_SCALE)
    parser.add_argument('--point-size',type=float,default=MARKER)
    args=parser.parse_args()
    if not (0<args.view_scale<=1 and 0<args.point_size<=40):
        raise ValueError('Require view scale in(0,1] and point size in(0,40]')
    OUT=args.output.resolve();VIEW_SCALE=args.view_scale;MARKER=args.point_size
    with threadpool_limits(limits=1):main()
