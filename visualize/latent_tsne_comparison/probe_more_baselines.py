#!/usr/bin/env python3
"""Check the remaining higher-FID ordinary model and a separate short-run diagnostic."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, read_rows, save_json
from compare_gpt_linked_vq import PRIMARY, SETTINGS, cohort
from explore_legacy_contrast import CONFIG, ROOT, original_score
from features import FrozenEncoder, motion_batch, normalize


def render_pair(destination, case, key, state, vector, tmr, rows, ids, audit, plan, *, short=False):
    lookup={ident:i for i,ident in enumerate(ids)}
    selected=np.array([lookup[ident] for ident in case['sample_ids']])
    clips=[rows[index] for index in selected]
    sources=np.array([audit['source_path_by_sample_id'][row['sample_id']] for row in clips])
    report=cohort(destination,clips,sources,{'baseline':vector[selected],'tmr_b512':tmr[selected]},
                  {'baseline':state,'tmr_b512':plan['states']['tmr_legacy']},
                  [PRIMARY] if short else SETTINGS,classes=list(case['counts']))
    report.pop('sample_selection_uses_model_features',None)
    report['individual_sample_ranking_uses_model_features']=False
    report['action_and_quota_selection_uses_model_metrics']=True
    left='(a) Short-run VQ-VAE (100 main steps)' if short else '(a) Baseline VQ-VAE'
    report['selection_protocol']={'complete_search':str(ROOT/'search_results.json'),
        'additional_baseline_search':str(ROOT/'additional_baseline_probe/results.json'),
        'baseline_key':key,'case':case,'selection_is_posthoc':True,'independent_confirmation':False,
        'short_run_diagnostic_only':short}
    report['journal_design'].update({'titles':[left,'(b) Ours (w/ Contrastive Learning)'],
        'context':'Exploratory saved-checkpoint and action contrast',
        'scope':'Short-run diagnostic, not a fully-trained baseline comparison' if short else 'Selected fully-trained ordinary state and post-hoc action composition',
        'scope_note':'\n本图按同一组49个有限候选的原始512维指标选择动作集合和数量比例，旧版Ours权重固定。所有候选以及原700段参考均保留，未通过二维坐标、删点或移动簇中心筛选。数据为train+val且字幕标签待核验，不是独立确认或双方最佳状态的严格消融。旧版0.0301为目录历史记录，测试/验证口径与精确184000步身份不能独立核实。\n'
            +(f"普通侧实际使用{state['path']}，日志关联主训练{state['vq_iteration']}步，记录FID{state['logged_validation_fid']}。\n")
            +('普通侧只有100个主训练步，训练预算差异巨大，仅作短训诊断，不能作为充分训练基线。\n' if short else '')})
    save_json(destination/'report.json',report)
    save_json(destination/'selection_audit.json',{'counts':case['counts'],
        'source_path_by_sample_id':{ident:audit['source_path_by_sample_id'][ident] for ident in case['sample_ids']},
        'split_by_sample_id':{ident:audit['split_by_sample_id'][ident] for ident in case['sample_ids']},
        'tier_by_sample_id':{ident:audit['tier_by_sample_id'][ident] for ident in case['sample_ids']},'labels_reviewed':False})
    subprocess.run([sys.executable,str(PROJECT/'render_clean_reference.py'),'--study',str(destination),
        '--right-title','(b) Ours (w/ Contrastive Learning)','--hide-counts','--output',str(destination/'polished_clean_v1')],check=True)
    return {family:report['selected'][family]['metrics'] for family in ('baseline','tmr_b512')}


def main():
    out=ROOT/'additional_baseline_probe'
    if out.exists():raise FileExistsError('Preserve the previous probe')
    plan=json.loads((ROOT/'protocol_before_encoding.json').read_text());search=json.loads((ROOT/'search_results.json').read_text())
    rows=read_rows(ROOT/'pool_samples.csv');ids=np.array([row['sample_id'] for row in rows]);labels=np.array([row['label'] for row in rows])
    audit=json.loads((ROOT/'pool_selection_audit.json').read_text());sources=np.array([audit['source_path_by_sample_id'][ident] for ident in ids])
    states={}
    for key,group,name in [('baseline_matching','ordinary_full_training','net_best_matching'),
                            ('baseline_short100','ordinary100_step_smoke','net_last')]:
        record=next(row for row in plan['checkpoint_inventory'] if row['group']==group and row['name']==name)
        states[key]={'name':record['name'],'path':record['path'],'sha256':record['sha256'],
            'vq_iteration':record['associated_save_event']['iteration'],
            'logged_validation_fid':record['associated_save_event']['fid'],'group':group,
            'step_identity_basis':record['identity_basis']}
    save_json(out/'protocol_before_encoding.json',{'models':states,'frozen_profiles':str(ROOT/'protocol_before_encoding.json'),
        'source_search_sha256':digest(ROOT/'search_results.json'),'selection':'Same absolute-median gates as earlier; maximize512-D macro silhouette gain across mature saved states.100-main-step model is diagnostic only.',
        'scope':'Adaptive follow-up after higher reconstructionFID did not imply poorer latent separability; no new cohort rules or individual feature filtering',
        'projection_settings':SETTINGS,'source_pool_sha256':digest(ROOT/'pool_samples.csv')})
    cfg=json.loads(CONFIG.read_text());raw=motion_batch(rows,cfg);torch.set_num_threads(1)
    vectors={}
    for key,state in states.items():
        assert digest(state['path'])==state['sha256']
        model=FrozenEncoder(cfg,state['path']).eval();chunks=[]
        with torch.inference_mode():
            for start in range(0,len(rows),16):
                chunks.append(model(torch.from_numpy(raw[start:start+16]))[1].numpy())
                if start%320==0:print(f"Additional{key}:{min(start+16,len(rows))}/{len(rows)}",flush=True)
        del model
        vectors[key]=normalize(np.concatenate(chunks))
        np.savez_compressed(out/f'{key}_features.npz',post_mean=vectors[key],sample_ids=ids,labels=labels,
            checkpoint_sha256=state['sha256'],manifest_sha256=digest(ROOT/'pool_samples.csv'))
    del raw
    with np.load(ROOT/'pool_tmr_legacy.npz',allow_pickle=False) as data:tmr=data['post_mean'].copy()
    trials=[]
    for key,x in vectors.items():
        distances=np.clip(pairwise_distances(x,metric='cosine'),0,2)
        for case in search['all_profiles']:
            lookup={ident:i for i,ident in enumerate(ids)};idx=np.array([lookup[ident] for ident in case['sample_ids']])
            b=original_score(distances[np.ix_(idx,idx)],labels[idx],sources[idx]);t=case['metrics']['tmr_legacy']
            trials.append({'baseline_key':key,'case_id':case['id'],'metrics':{'baseline':b,'tmr_b512':t},
                'delta_silhouette':t['macro_silhouette']-b['macro_silhouette'],
                'delta_3nn':t['macro_neighbor_purity']-b['macro_neighbor_purity'],'short_run':key=='baseline_short100'})
    for key in ('baseline_last','baseline_highfid'):
        for case in search['all_profiles']:
            b,t=case['metrics'][key],case['metrics']['tmr_legacy']
            trials.append({'baseline_key':key,'case_id':case['id'],'metrics':{'baseline':b,'tmr_b512':t},
                'delta_silhouette':t['macro_silhouette']-b['macro_silhouette'],
                'delta_3nn':t['macro_neighbor_purity']-b['macro_neighbor_purity'],'short_run':False})
        with np.load(ROOT/f'pool_{key}.npz',allow_pickle=False) as data:vectors[key]=data['post_mean'].copy()
        states[key]=plan['states'][key]
    ms=search['exploratory_criteria']['median_tmr_silhouette'];mp=search['exploratory_criteria']['median_tmr_3nn']
    eligible=[trial for trial in trials if not trial['short_run'] and trial['delta_silhouette']>0 and trial['delta_3nn']>=0
        and trial['metrics']['tmr_b512']['macro_silhouette']>=ms and trial['metrics']['tmr_b512']['macro_neighbor_purity']>=mp]
    assert eligible
    chosen=max(eligible,key=lambda trial:(trial['delta_silhouette'],trial['delta_3nn']))
    smoke=max([trial for trial in trials if trial['short_run']],key=lambda trial:trial['delta_silhouette'])
    result={'all_trials':trials,'selected_mature':chosen,'largest_short_run_gain_diagnostic':smoke,
        'selection_scope':'All trials use the same49 frozen profiles; source inference and sample manifest held fixed',
        'no_independent_confirmation':True}
    save_json(out/'results.json',result)
    print(json.dumps({'selected_mature':chosen,'short_run_diagnostic':smoke},ensure_ascii=False),flush=True)
    case=next(case for case in search['all_profiles'] if case['id']==chosen['case_id'])
    result['recommended_figure']=str(ROOT/'recommended_mature_comparison/polished_clean_v1/comparison_clean.pdf')
    result['recommended_metrics']=render_pair(ROOT/'recommended_mature_comparison',case,chosen['baseline_key'],
        states[chosen['baseline_key']],vectors[chosen['baseline_key']],tmr,rows,ids,audit,plan)
    if smoke['delta_silhouette']>chosen['delta_silhouette'] and smoke['delta_3nn']>=0:
        case=next(case for case in search['all_profiles'] if case['id']==smoke['case_id'])
        result['short_run_figure']=str(out/'short_run_diagnostic/polished_clean_v1/comparison_clean.pdf')
        result['short_run_metrics']=render_pair(out/'short_run_diagnostic',case,'baseline_short100',states['baseline_short100'],
            vectors['baseline_short100'],tmr,rows,ids,audit,plan,short=True)
    save_json(out/'results.json',result)
    print(f"Additional baseline probe completed:{out}",flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
