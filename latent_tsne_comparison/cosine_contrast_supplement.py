#!/usr/bin/env python3
"""An explicitly separate cosine t-SNE view of the same selected motion features."""
import copy
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from common import PROJECT,digest,read_rows,save_json,write_rows
from explore_legacy_contrast import ROOT
from optimize_display import fit_cosine,score
from render_polished import PALETTE


def main():
    out=ROOT/'recommended_cosine_supplement'
    if out.exists():raise FileExistsError('Preserve previous projections')
    plan=json.loads((ROOT/'protocol_before_encoding.json').read_text())
    search=json.loads((ROOT/'search_results.json').read_text())
    probe=json.loads((ROOT/'additional_baseline_probe/results.json').read_text())
    chosen=probe['selected_mature'];case=next(c for c in search['all_profiles'] if c['id']==chosen['case_id'])
    baseline_key=chosen['baseline_key']
    if baseline_key not in plan['states']:
        raise ValueError('This supplemental command expects a baseline from the primary cached pool')
    rows=read_rows(ROOT/'pool_samples.csv');lookup={row['sample_id']:i for i,row in enumerate(rows)}
    idx=np.array([lookup[ident] for ident in case['sample_ids']]);clips=[rows[i] for i in idx]
    ids=np.array(case['sample_ids']);labels=np.array([row['label'] for row in clips])
    audit=json.loads((ROOT/'pool_selection_audit.json').read_text());sources=np.array([audit['source_path_by_sample_id'][ident] for ident in ids])
    states={'baseline':plan['states'][baseline_key],'tmr_b512':plan['states']['tmr_legacy']}
    method={'metric':'cosine','PCA':False,'perplexities':[20,30,50],'seed':42,'early_exaggeration':12,
        'iterations':2000,'primary_perplexity':30,'same_settings_on_both_models':True,
        'purpose':'Alternative display of unchanged512-D features; not a new representation or training gain'}
    save_json(out/'protocol_before_projection.json',{'case':case,'states':states,'projection':method,
        'selection_source':str(ROOT/'additional_baseline_probe/results.json'),
        'scope':'Same post-hoc action composition and saved model states; no independent confirmation'})
    write_rows(out/'samples.csv',clips)
    fits={};records=[];selected={}
    for family,key in [('baseline',baseline_key),('tmr_b512','tmr_legacy')]:
        with np.load(ROOT/f'pool_{key}.npz',allow_pickle=False) as data:
            assert str(data['checkpoint_sha256'])==states[family]['sha256'];x=data['post_mean'][idx].copy()
        np.savez_compressed(out/f'{family}_features.npz',post_mean=x,sample_ids=ids,labels=labels,
            checkpoint_sha256=states[family]['sha256'])
        metrics=score(x,labels,sources)
        for metric in ['macro_silhouette','macro_neighbor_purity']:
            np.testing.assert_allclose(metrics[metric],chosen['metrics'][family][metric],atol=1e-7)
        selected[family]={**states[family],'metrics':metrics}
        fits[family]={}
        for p in method['perplexities']:
            xy,kl,trust=fit_cosine(x,p,42);fits[family][p]=xy
            np.savez_compressed(out/f'{family}_cosine_p{p}_s42.npz',coordinates=xy,sample_ids=ids,labels=labels,
                kl_divergence=kl,features_sha256=digest(out/f'{family}_features.npz'),trustworthiness_original_cosine_k10=trust)
            records.append({'family':family,'perplexity':p,'seed':42,'kl_divergence':kl,'trustworthiness_original_cosine_k10':trust})
            print(f"Cosine supplement:{family},P{p},trust{trust:.4f}",flush=True)
    report={'samples':len(clips),'classes':len(case['counts']),'class_counts':case['counts'],'selected':selected,
        'individual_sample_ranking_uses_model_features':False,'action_and_quota_selection_uses_model_metrics':True,
        'split':'train+val','labels_reviewed':False,'independent_test':False,
        'selection_protocol':{'complete_search':str(ROOT/'search_results.json'),'case':case,'projection':method,
            'projection_is_supplement_not_new_model_evidence':True},
        'journal_design':{'titles':['(a) Baseline VQ-VAE','(b) Ours (w/ Contrastive Learning)'],
            'coordinate_files':{family:f'{family}_cosine_p30_s42.npz' for family in states},
            'projection_caption':'Cosine t-SNE is fitted directly on the original features, without PCA, independently for each model with identical settings (perplexity30, early exaggeration12, seed42,2000 iterations). ',
            'scope':'Cosine t-SNE supplement; unchanged model states and finite-search-selected action composition',
            'scope_note':'\n本图使用与推荐主图相同的700段、6动作、权重与原始512维特征，只将PCA50→欧氏t-SNE替换为原始512维cosine t-SNE。两模型共享P30/E12/seed42及2000次迭代，另保留P20/P50图。指标仍在原始512维计算，不能把投影变化解释为模型改善。动作/配额与权重经过事后筛选；train+val、字幕标签待核验、旧版0.0301与184000步身份和评估口径限制见来源报告。\n'}}
    save_json(out/'report.json',report);save_json(out/'fits.json',records)
    fig,axes=plt.subplots(3,2,figsize=(9,9))
    for row,p in enumerate(method['perplexities']):
        for col,family in enumerate(states):
            ax=axes[row,col];xy=fits[family][p]
            for label in case['counts']:
                points=xy[labels==label];ax.scatter(*points.T,s=7,color=PALETTE[label],alpha=.9,linewidths=0)
            ax.set(xticks=[],yticks=[]);ax.set_aspect('equal',adjustable='datalim')
            ax.set_title(f"{'Baseline' if col==0 else 'Ours'} | cosine P{p},seed42")
    fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=180);fig.savefig(out/'sensitivity.pdf');plt.close(fig)
    subprocess.run([sys.executable,str(PROJECT/'render_clean_reference.py'),'--study',str(out),
        '--right-title','(b) Ours (w/ Contrastive Learning)','--hide-counts','--output',str(out/'polished_clean_v1')],check=True)
    print(f"Cosine supplement completed:{out}",flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
