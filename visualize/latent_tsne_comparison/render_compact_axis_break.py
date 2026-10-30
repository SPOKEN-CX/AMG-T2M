#!/usr/bin/env python3
"""Compact a genuinely empty x interval with an explicitly marked broken axis.
No features, samples, model parameters or high-dimensional metrics are modified.
Across the marked break, display distances must not be interpreted quantitatively.
"""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import pdist
from common import PROJECT,digest,save_json
from render_semantic_experiment import panel,metric
from render_polished import PALETTE

SOURCE=PROJECT/'artifacts/semantic_metric_head_v1/fusion_soft_v1/previous_exploratory420'
OUT=SOURCE/'compact_axis_break_v1'


def main():
    if OUT.exists(): raise FileExistsError('Preserve previous exports; choose a new output directory')
    input_csv=SOURCE/'real_fusion_comparison_plotted_coordinates.csv'
    manifest_path=SOURCE/'real_fusion_comparison_manifest.json'
    original=json.loads(manifest_path.read_text())
    with input_csv.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    labels=np.array([r['label'] for r in rows]);ids=np.array([r['sample_id'] for r in rows]);classes=list(dict.fromkeys(labels))
    left=np.array([[float(r['left_x']),float(r['left_y'])] for r in rows])
    right=np.array([[float(r['right_x']),float(r['right_y'])] for r in rows])
    order=np.random.default_rng(42).permutation(len(rows))
    # The user indicated the empty strip before the far-right sit component.
    # Find that strip from x coordinates alone; no class labels enter the rule.
    sx=np.sort(right[:,0]);gaps=np.diff(sx);j=int(np.argmax(gaps));lo,hi=float(sx[j]),float(sx[j+1])
    width=hi-lo;omit_fraction=.60
    cut_lo=lo+(1-omit_fraction)*width/2;cut_hi=hi-(1-omit_fraction)*width/2
    delta=cut_hi-cut_lo
    assert not ((right[:,0]>cut_lo)&(right[:,0]<cut_hi)).any()
    is_right=right[:,0]>=cut_hi
    shown=right.copy();shown[is_right,0]-=delta
    for mask in [~is_right,is_right]:
        np.testing.assert_allclose(pdist(right[mask]),pdist(shown[mask]),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(shown[:,1],right[:,1])
    # Freeze the display-only protocol before writing the plot.
    OUT.mkdir()
    save_json(OUT/'display_protocol.json',{
        'operation':'explicitly marked x-axis break removes part of a point-free interval',
        'display_break_is_not_a_model_improvement':True,'input_sha256':{str(p):digest(p) for p in [input_csv,manifest_path]},
        'original_empty_interval':[lo,hi],'omitted_interval':[cut_lo,cut_hi],
        'fraction_of_circled_empty_interval_omitted':omit_fraction,
        'points_deleted':0,'points_added':0,'class_labels_used_to_choose_or_apply_break':False,
        'features_models_and_metrics_unchanged':True,
        'within_each_unbroken_segment_geometry_preserved':True,
        'distances_across_marked_break_are_not_quantitative':True,
        'right_title':'(b) Ours (w/ Contrastive Learning)',
        'method_scope':'new caption-supervised head plus soft motion-semantic fusion, not unchanged raw VQ; original scientific qualifications retained in caption'})

    plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'text.color':'#26313D','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(8.4,3.05),facecolor='white')
    for pos,xy,title,key in [(.022,left,'(a) Baseline VQ-VAE','baseline_raw'),(.440,shown,'(b) Ours (w/ Contrastive Learning)','ours_fusion_soft')]:
        ax=fig.add_axes((pos,.205,.4,.67));panel(ax,xy,labels,order)
        fig.text(pos+.2,.951,title,ha='center',va='center',fontsize=11,fontweight='bold')
        metric(fig,pos+.2,original['metrics'][key],512 if key=='baseline_raw' else 640)
        if key=='ours_fusion_soft':
            right_ax=ax
    fig.canvas.draw()
    # Broken-axis markers remain inside the scientific panel and in every export.
    display_x=right_ax.transData.transform((cut_lo,0))[0]
    seam=right_ax.transAxes.inverted().transform((display_x,0))[0]
    assert 0<seam<1
    right_ax.axvline(cut_lo,color='#D3D9DF',lw=.45,ls=(0,(2,4)),zorder=1)
    for y in [0,1]:
        right_ax.plot([seam-.017,seam+.017],[y,y],transform=right_ax.transAxes,c='white',lw=3,clip_on=False,zorder=7)
        for dx in [-.006,.006]:
            right_ax.plot([seam+dx-.004,seam+dx+.004],[y-.012,y+.012],transform=right_ax.transAxes,c='#5B6773',lw=.85,clip_on=False,zorder=8)
    right_ax.text(seam,.945,'x-axis break',transform=right_ax.transAxes,ha='center',va='center',fontsize=6.4,color='#5B6773',
        bbox={'facecolor':'white','edgecolor':'none','pad':1.1,'alpha':.95},zorder=9)

    leg=fig.add_axes((.858,.205,.122,.670));leg.set_axis_off()
    leg.add_patch(FancyBboxPatch((0,0),1,1,transform=leg.transAxes,boxstyle='round,pad=0,rounding_size=.024',edgecolor='#D2D8DE',facecolor='#FCFDFE',linewidth=.6,clip_on=False))
    leg.text(.5,.937,'Action',fontsize=9.4,fontweight='bold',va='center',ha='center');leg.plot([.11,.89],[.885,.885],color='#E0E5E9',lw=.55)
    for label,y in zip(classes,np.linspace(.808,.103,len(classes))):
        leg.scatter([.2],[y],s=24,c=PALETTE[label],edgecolors='white',linewidths=.3);leg.text(.37,y,label,fontsize=9.1,va='center')
    leg.set(xlim=(0,1),ylim=(0,1))
    fig.text(.5,.041,'Right panel: marked x-axis break omits an empty interval; cross-break display distances are not quantitative.',ha='center',va='center',fontsize=5.8,color='#657180')
    fig.text(.5,.015,'Exploratory420 clips;296 fit recordings. Caption-supervised head + soft fusion; labels await review.',ha='center',va='center',fontsize=5.8,color='#657180')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax,xy in [(fig.axes[0],left),(right_ax,shown)]:
        bounds=ax.get_window_extent(renderer);pixels=ax.transData.transform(xy)
        assert (pixels[:,0]>bounds.x0+1).all() and (pixels[:,0]<bounds.x1-1).all()
        assert (pixels[:,1]>bounds.y0+1).all() and (pixels[:,1]<bounds.y1-1).all()
        a,b,c=ax.transData.transform([[0,0],[1,0],[0,1]])
        np.testing.assert_allclose(np.linalg.norm(a-b),np.linalg.norm(a-c),rtol=1e-6)
    for ext in ['pdf','svg']:fig.savefig(OUT/f'comparison_compact.{ext}')
    fig.savefig(OUT/'comparison_compact.png',dpi=900);fig.savefig(OUT/'comparison_compact_preview.png',dpi=300);plt.close(fig)
    with (OUT/'original_and_display_coordinates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','label','baseline_x','baseline_y','ours_original_x','ours_original_y','ours_display_x','ours_display_y','break_segment'])
        for i,r in enumerate(rows):w.writerow([r['sample_id'],r['label'],*left[i],*right[i],*shown[i],int(is_right[i])])
    caption=(SOURCE/'real_fusion_comparison_CAPTION.txt').read_text()+(
        '\nDISPLAY-ONLY COMPACT VERSION: The right-panel title is Ours (w/ Contrastive Learning). '
        'It still denotes the NEW caption-supervised semantic-head + soft fusion representation specified above. '
        f'An explicitly marked broken x axis omits the point-free interval[{cut_lo:.6f},{cut_hi:.6f}] of the rotated t-SNE coordinates. '
        'This removes60% of the largest empty horizontal gap. All420 points are retained. '
        'All coordinates and pairwise geometry within each unbroken segment are preserved up to a horizontal translation; '
        'cross-break visual distances are intentionally compressed and must not be interpreted as smaller latent distances or improved separation. '
        'Displayed metrics remain the unchanged original high-dimensional metrics. The full unbroken plot is retained in the parent directory.\n')
    (OUT/'FIGURE_CAPTION.txt').write_text(caption,encoding='utf-8')
    (OUT/'README.zh-CN.md').write_text(
        '# 明确断轴的紧凑版\n\n按用户圈定的空白区间，省略最大空白横向间隔中央60%，并在右图保留断轴斜线、竖向提示与x-axis break标签。'
        '未删除/复制/合成数据点；区间两侧分别保持原始局部几何；跨断轴的展示距离被压缩，因此不能据此声称真实潜空间距离变短。'
        '两个面板外框尺寸、点样式、共享图例和原高维指标一致，整图2520×915预览，900dpi PNG为7560×2745。\n\n'
        '标题按要求改为(b) Ours (w/ Contrastive Learning)，但方法依旧是新增监督语义分支+软融合，不能改称未修改的原VQ潜空间。'
        '420图有296个录制来源参与分支拟合，弱标签待人工核验；完整实验资格说明保留于FIGURE_CAPTION.txt。\n',encoding='utf-8')
    save_json(OUT/'export_manifest.json',{'samples_per_panel':len(rows),'class_counts':dict(Counter(labels)),
        'display_axis_break':True,'omitted_fraction_of_empty_gap':omit_fraction,
        'unchanged_metrics':original['metrics'],'right_title':'(b) Ours (w/ Contrastive Learning)',
        'feature_change':False,'model_change':False,'all_points_retained':True,
        'both_panels_isotropic_within_segments':True,'pixels':[7560,2745],'preview_pixels':[2520,915],
        'output_sha256':{p.name:digest(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'output':str(OUT),'empty_gap_reduced_by':omit_fraction,'omitted_x_span':delta,'title':'(b) Ours (w/ Contrastive Learning)'},ensure_ascii=False))

if __name__=='__main__':main()
