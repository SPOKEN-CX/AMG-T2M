#!/usr/bin/env python3
"""Finite, disclosed exploration of saved ordinary checkpoints and action proportions."""
import argparse
import copy
import itertools
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import pairwise_distances, silhouette_samples
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, read_rows, save_json, write_rows
from compare_gpt_linked_vq import PRIMARY, SETTINGS, cohort
from features import FrozenEncoder, motion_batch, normalize
from pilot_actions700 import ranked_indices


POOL_CLASSES = ["run", "jump", "sit", "kick", "throw", "dance", "wave", "walk", "arm_raise"]
ROOT = PROJECT / "artifacts/legacy_contrast_exploration_v1"
SOURCE = PROJECT / "artifacts/all_fid_audit_700_v1/legacy_lowest_record_vs_netlast_700"
CATALOG = PROJECT / "artifacts/action_search_700_merged_v1"
CONFIG = PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json"


def checkpoint_inventory():
    records = []
    for directory, group in [(REPO / "output/hml3d/VQ_base", "ordinary_full_training"),
                             (REPO / "output/hml3d/VQ_check100", "ordinary100_step_smoke"),
                             (REPO / "pretrained/VQVAE", "different_pretrained_run")]:
        current, latest = None, {}
        for number, line in enumerate((directory / "run.log").read_text().splitlines(), 1):
            match = re.search(r"Eva\. Iter (\d+)\s*:, FID\. ([\d.eE+-]+)", line)
            if match:
                current = {"iteration": int(match[1]), "fid": float(match[2]),
                           "line": number, "text": line}
            for name, trigger in [("net_best_fid", "FID Improved"), ("net_best_div", "Diversity Improved"),
                                  ("net_best_top1", "Top1 Improved"), ("net_best_matching", "matching_score Improved")]:
                if trigger in line and current:
                    latest[name] = current.copy()
        latest["net_last"] = current
        for path in sorted(directory.glob("*.pth")):
            records.append({"name": path.stem, "path": str(path), "sha256": digest(path),
                "group": group, "associated_save_event": latest.get(path.stem),
                "identity_basis": "Existing filename and final corresponding save trigger in its run.log; no numbered ordinary checkpoint archive"})
    return records


def equal_allocation(classes, available, target=700, ratio=1.5):
    counts = {label: min(available[label], target // len(classes)) for label in classes}
    while sum(counts.values()) < target:
        possible = [label for label in classes if counts[label] < available[label]]
        if not possible:
            return None
        label = min(possible, key=lambda key: (counts[key], classes.index(key)))
        counts[label] += 1
    return counts if max(counts.values()) / min(counts.values()) <= ratio else None


def make_profiles(available):
    profiles = [
        {"id": "original700", "counts": {"run":112, "jump":111, "sit":81, "kick":75, "throw":98, "dance":112, "wave":111},
         "scope": "Fixed prior700 control; only the ordinary checkpoint changes"},
        {"id": "jump_weighted700", "counts": {"run":110, "jump":135, "sit":81, "kick":75, "throw":98, "dance":90, "wave":111},
         "scope": "Same seven actions; more jumping and fewer heterogeneous dance clips; ratio1.8 exploratory profile"},
        {"id": "six_without_dance700", "counts": {"run":149, "jump":149, "sit":81, "kick":75, "throw":98, "wave":148},
         "scope": "Six actions; remove heterogeneous dance; max/min1.987; prior per-action results informed this profile"},
        {"id": "six_without_wave682", "counts": {"run":150, "jump":150, "sit":81, "kick":75, "throw":98, "dance":128},
         "scope": "Six actions; remove prior negative-gain wave category;682 clips, ratio2; explicitly post-hoc"},
        {"id": "replace_dance_walk700", "counts": {"run":112, "jump":111, "sit":81, "kick":75, "throw":98, "walk":112, "wave":111},
         "scope": "Replace dance with walk at the same quota; ratio1.493"},
        {"id": "replace_dance_arm_raise700", "counts": {"run":112, "jump":111, "sit":81, "kick":75, "throw":98, "arm_raise":112, "wave":111},
         "scope": "Replace dance with broader arm_raise at the same quota; semantic specificity is lower; ratio1.493"},
    ]
    seen = {tuple(sorted(profile["counts"].items())) for profile in profiles}
    for k in (6, 7):
        for chosen in itertools.combinations(POOL_CLASSES, k):
            counts = equal_allocation(list(chosen), available)
            if counts is None or tuple(sorted(counts.items())) in seen:
                continue
            seen.add(tuple(sorted(counts.items())))
            profiles.append({"id": f"balanced{k}_{len(profiles):02d}", "counts": counts,
                "scope": "Enumerated700-clip,6/7-action composition with max/min<=1.5; whole category selection is exploratory"})
    for profile in profiles:
        counts = profile["counts"]
        assert all(2 <= n <= available[label] for label, n in counts.items())
        profile["samples"] = sum(counts.values())
        profile["max_min_ratio"] = max(counts.values()) / min(counts.values())
        profile["strict700"] = profile["samples"] == 700 and profile["max_min_ratio"] <= 1.5
    return profiles


def encode(out):
    if out.exists():
        raise FileExistsError("Choose a new output directory; previous searches are preserved")
    candidates = read_rows(CATALOG / "candidates.csv")
    audit = json.loads((CATALOG / "selection_audit.json").read_text())
    prior = json.loads((SOURCE / "report.json").read_text())
    inventory = checkpoint_inventory()
    eligible = [record for record in inventory if record["group"] == "ordinary_full_training"]
    highfid = max(eligible, key=lambda record: record["associated_save_event"]["fid"])
    assert highfid["name"] == "net_best_top1" and highfid["associated_save_event"]["iteration"] == 94000
    indices = np.concatenate([ranked_indices(candidates, label, audit)[:160] for label in POOL_CLASSES])
    rows = [candidates[index] for index in indices]
    ids = np.array([row["sample_id"] for row in rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    assert len(rows) == len(set(ids)) == len(set(sources))
    available = dict(Counter(row["label"] for row in rows))
    states = {"baseline_last": {k:v for k,v in prior["selected"]["baseline"].items() if k != "metrics"},
              "baseline_highfid": {"name": highfid["name"], "path": highfid["path"], "sha256": highfid["sha256"],
                 "vq_iteration": highfid["associated_save_event"]["iteration"],
                 "logged_validation_fid": highfid["associated_save_event"]["fid"],
                 "step_identity_basis": "filename/log-save association; no independent numbered checkpoint"},
              "tmr_legacy": {k:v for k,v in prior["selected"]["tmr_b512"].items() if k != "metrics"}}
    plan = {"scope": "User-authorized selection of higher-FID ordinary state and post-hoc action proportions; no independent confirmation",
        "checkpoint_inventory": inventory, "states": states,
        "baseline_selection": "Highest logged reconstruction FID among stored states of the same fully-trained VQ_base run; excludes100-step smoke models",
        "source_legacy_identity_limit": prior["selected"]["tmr_b512"],
        "pool_classes": POOL_CLASSES, "pool_counts": available, "pool_samples": len(rows),
        "pool_selection": "Caption-evidence tier, then unchanged SHA256(seed20261030:sample_id); first160 per category; no individual outcome-based filtering",
        "profiles_before_encoding": make_profiles(available),
        "exploration_selection": "For strict700 and all-profile groups separately: positive silhouette gain and nonnegative3NN gain; TMR absolute silhouette/purity at least group medians; maximize original512-D macro silhouette gain, tie-break3NN gain",
        "display_settings": {"main": list(PRIMARY), "winner_sensitivity": SETTINGS,
            "pca":50, "metric":"euclidean", "init":"pca", "max_iter":3000},
        "selection_uses_2D_coordinates": False,
        "no_new_training": True, "teacher_features_used": False, "no_external_data": True,
        "feature": "L2-normalized512-D post-quantization temporal mean; same64-frame crop and normalization",
        "labels_reviewed": False, "data_split": "train+val", "independent_test": False,
        "input_sha256": {str(p):digest(p) for p in [CATALOG/'candidates.csv',CATALOG/'selection_audit.json',
            CONFIG,SOURCE/'samples.csv',SOURCE/'report.json',SOURCE/'baseline_features.npz',SOURCE/'tmr_b512_features.npz',
            REPO/'output/hml3d/VQ_base/run.log', REPO/'output/hml3d/VQ_check100/run.log', REPO/'pretrained/VQVAE/run.log']}}
    save_json(out / "protocol_before_encoding.json", plan)
    write_rows(out / "pool_samples.csv", rows)
    save_json(out / "pool_selection_audit.json", {
        "source_path_by_sample_id": dict(zip(ids, sources)),
        "split_by_sample_id": {ident:audit["split_by_sample_id"][ident] for ident in ids},
        "tier_by_sample_id": {ident:audit["tier_by_sample_id"][ident] for ident in ids},
        "labels_pending_review": True})
    cfg = json.loads(CONFIG.read_text())
    for row in rows:
        assert digest(REPO / cfg["dataset_root"] / "new_joint_vecs" / f"{row['motion_id']}.npy") == row["motion_sha256"]
        assert int(row["end_frame"])-int(row["start_frame"]) ==64
    torch.set_num_threads(1)
    labels = np.array([row["label"] for row in rows])
    for family, state in states.items():
        assert digest(state["path"]) == state["sha256"]
        features = np.empty((len(rows),512),dtype=np.float32)
        missing = np.ones(len(rows),dtype=bool)
        if family in ("baseline_last", "tmr_legacy"):
            filename = "baseline_features.npz" if family == "baseline_last" else "tmr_b512_features.npz"
            with np.load(SOURCE / filename,allow_pickle=False) as cached:
                assert str(cached["checkpoint_sha256"]) == state["sha256"]
                lookup = {ident:i for i,ident in enumerate(cached["sample_ids"])}
                for index, ident in enumerate(ids):
                    if ident in lookup:
                        features[index] = cached["post_mean"][lookup[ident]]
                        missing[index] = False
        selected = np.flatnonzero(missing)
        raw = motion_batch([rows[index] for index in selected],cfg)
        model = FrozenEncoder(cfg,state["path"]).eval()
        with torch.inference_mode():
            for start in range(0,len(selected),16):
                post = model(torch.from_numpy(raw[start:start+16]))[1].numpy()
                features[selected[start:start+16]] = normalize(post)
                if start%160==0:
                    print(f"{family}: encoding{min(start+16,len(selected))}/{len(selected)}, reusing{int((~missing).sum())}",flush=True)
        del model,raw
        assert np.isfinite(features).all()
        np.testing.assert_allclose(np.linalg.norm(features,axis=1),1,atol=1e-6)
        np.savez_compressed(out / f"pool_{family}.npz",post_mean=features,sample_ids=ids,labels=labels,
                            checkpoint_sha256=state["sha256"],manifest_sha256=digest(out/'pool_samples.csv'))
    print(f"Encoded finite pool:{len(rows)} records,{len(plan['profiles_before_encoding'])} planned profiles",flush=True)


def original_score(distances, labels, sources):
    distances = distances.copy()
    np.fill_diagonal(distances,0)
    silhouettes = silhouette_samples(distances,labels,metric="precomputed")
    distances[sources[:,None]==sources[None,:]]=np.inf
    nearest=np.argsort(distances,axis=1,kind="stable")[:,:3]
    purity=(labels[nearest]==labels[:,None]).mean(1)
    per_class={label:{"samples":int((labels==label).sum()),
               "silhouette":float(silhouettes[labels==label].mean()),
               "neighbor_purity":float(purity[labels==label].mean())} for label in dict.fromkeys(labels)}
    return {"macro_silhouette":float(np.mean([row['silhouette'] for row in per_class.values()])),
            "macro_neighbor_purity":float(np.mean([row['neighbor_purity'] for row in per_class.values()])),
            "per_class":per_class}


def select_case(cases):
    median_s=float(np.median([case['metrics']['tmr_legacy']['macro_silhouette'] for case in cases]))
    median_p=float(np.median([case['metrics']['tmr_legacy']['macro_neighbor_purity'] for case in cases]))
    eligible=[case for case in cases if case['delta_silhouette']>0 and case['delta_3nn']>=0
              and case['metrics']['tmr_legacy']['macro_silhouette']>=median_s
              and case['metrics']['tmr_legacy']['macro_neighbor_purity']>=median_p]
    return (max(eligible,key=lambda case:(case['delta_silhouette'],case['delta_3nn'])) if eligible else None,
            {"median_tmr_silhouette":median_s,"median_tmr_3nn":median_p,"eligible":len(eligible)})


def analyze(out):
    plan=json.loads((out/'protocol_before_encoding.json').read_text())
    rows=read_rows(out/'pool_samples.csv');labels=np.array([row['label'] for row in rows]);ids=np.array([row['sample_id'] for row in rows])
    audit=json.loads((out/'pool_selection_audit.json').read_text());sources=np.array([audit['source_path_by_sample_id'][ident] for ident in ids])
    distances={}
    for family in plan['states']:
        with np.load(out/f'pool_{family}.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['sample_ids'],ids)
            assert str(data['checkpoint_sha256'])==plan['states'][family]['sha256']
            distances[family]=np.clip(pairwise_distances(data['post_mean'],metric='cosine'),0,2)
    cases=[]
    for profile in plan['profiles_before_encoding']:
        idx=np.concatenate([np.flatnonzero(labels==label)[:n] for label,n in profile['counts'].items()])
        metrics={family:original_score(d[np.ix_(idx,idx)],labels[idx],sources[idx]) for family,d in distances.items()}
        b,t=metrics['baseline_highfid'],metrics['tmr_legacy']
        cases.append({**profile,'metrics':metrics,'sample_ids':ids[idx].tolist(),
            'delta_silhouette':t['macro_silhouette']-b['macro_silhouette'],
            'delta_3nn':t['macro_neighbor_purity']-b['macro_neighbor_purity'],
            'delta_silhouette_vs_netlast':t['macro_silhouette']-metrics['baseline_last']['macro_silhouette']})
    strict,criteria_s=select_case([case for case in cases if case['strict700']])
    exploratory,criteria_e=select_case(cases)
    report={'scope':plan['scope'],'all_profiles':cases,'selected_strict700':strict,
            'selected_exploratory':exploratory,'strict_criteria':criteria_s,'exploratory_criteria':criteria_e,
            'selection_uses_original512D_not_2D':True,'independent_confirmation':False,
            'original700':next(case for case in cases if case['id']=='original700')}
    save_json(out/'search_results.json',report)
    import csv
    with (out/'all_profiles.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['profile','samples','max_min_ratio','strict700','class_counts',
            'baseline_highfid_silhouette','tmr_silhouette','delta_silhouette','baseline_highfid_3nn','tmr_3nn','delta3nn_pp',
            'netlast_silhouette','delta_silhouette_vs_netlast'])
        for case in sorted(cases,key=lambda item:-item['delta_silhouette']):
            b,t=case['metrics']['baseline_highfid'],case['metrics']['tmr_legacy']
            writer.writerow([case['id'],case['samples'],case['max_min_ratio'],case['strict700'],json.dumps(case['counts']),
                b['macro_silhouette'],t['macro_silhouette'],case['delta_silhouette'],b['macro_neighbor_purity'],t['macro_neighbor_purity'],
                100*case['delta_3nn'],case['metrics']['baseline_last']['macro_silhouette'],case['delta_silhouette_vs_netlast']])
    print(json.dumps({key:({name:value for name,value in case.items() if name not in ['sample_ids','metrics']} if case else None)
                     for key,case in [('strict700',strict),('exploratory',exploratory)]},ensure_ascii=False,indent=2),flush=True)
    control=report['original700']
    print(json.dumps({'original700_all_models':control['metrics']},ensure_ascii=False),flush=True)


def render(out):
    plan=json.loads((out/'protocol_before_encoding.json').read_text());search=json.loads((out/'search_results.json').read_text())
    rows=read_rows(out/'pool_samples.csv');ids=np.array([row['sample_id'] for row in rows]);lookup={ident:i for i,ident in enumerate(ids)}
    audit=json.loads((out/'pool_selection_audit.json').read_text());vectors={}
    for family in plan['states']:
        with np.load(out/f'pool_{family}.npz',allow_pickle=False) as data:vectors[family]=data['post_mean'].copy()
    chosen={case['id']:case for case in [search['original700'],search['selected_strict700'],search['selected_exploratory']] if case}
    summaries=[]
    for name,case in chosen.items():
        study=out/name
        if (study/'polished_clean_v1/render_manifest.json').exists():
            print(f"Preserving completed{study}",flush=True);continue
        selected=np.array([lookup[ident] for ident in case['sample_ids']]);clips=[rows[index] for index in selected]
        sources=np.array([audit['source_path_by_sample_id'][row['sample_id']] for row in clips])
        full=search['selected_exploratory'] is not None and name==search['selected_exploratory']['id']
        states={'baseline':plan['states']['baseline_highfid'],'tmr_b512':plan['states']['tmr_legacy']}
        data={'baseline':vectors['baseline_highfid'][selected],'tmr_b512':vectors['tmr_legacy'][selected]}
        report=cohort(study,clips,sources,data,states,SETTINGS if full else [PRIMARY],classes=list(case['counts']))
        report.pop('sample_selection_uses_model_features',None)
        report['individual_sample_ranking_uses_model_features']=False
        report['action_and_quota_selection_uses_model_metrics']=True
        report['selection_protocol']={'search_plan':str(out/'protocol_before_encoding.json'),
            'complete_profile_results':str(out/'search_results.json'),'profile':case,
            'selection_is_posthoc':True,'no_independent_confirmation':True,
            'ordinary_checkpoint_selection':'Higher reconstructionFID savedstate of same fully-trained run; not worst possible ordinary model',
            'legacy_exact184000_identity_unverified':True}
        report['journal_design'].update({'titles':['(a) Baseline VQ-VAE','(b) Ours (w/ Contrastive Learning)'],
            'context':f"Selected checkpoint/action contrast |{len(clips)} clips",
            'subtitles':{'baseline':'net_best_top1 | log-associated VQ step94,000 | validationFID0.1763',
                         'tmr_b512':'Legacy net_best_fid | historical directory minimumFID0.0301'},
            'scope':'Exploratory higher-FID ordinary checkpoint and post-hoc action-composition comparison',
            'scope_note':'\n普通侧按同一完整训练实验已保存权重中的更高重建FID选择net_best_top1（日志94000步、0.1763），保留原700段net_last参考。动作集合和比例从预先列出的有限候选中按原始512维分数选择，全部候选保留，未经独立验证。旧版目录日志最低0.0301，可用旧源码指向测试集；权重仅含net，不能独立核验184000步身份。\n'})
        save_json(study/'report.json',report)
        save_json(study/'selection_audit.json',{'counts':case['counts'],'source_path_by_sample_id':dict(zip(case['sample_ids'],sources)),
            'split_by_sample_id':{ident:audit['split_by_sample_id'][ident] for ident in case['sample_ids']},
            'tier_by_sample_id':{ident:audit['tier_by_sample_id'][ident] for ident in case['sample_ids']},'labels_reviewed':False})
        subprocess.run([sys.executable,str(PROJECT/'render_clean_reference.py'),'--study',str(study),
                        '--right-title','(b) Ours (w/ Contrastive Learning)','--hide-counts',
                        '--output',str(study/'polished_clean_v1')],check=True)
        summaries.append({'profile':name,'scope':case['scope'],'counts':case['counts'],'samples':len(clips),
            'figure':str(study/'polished_clean_v1/comparison_clean.pdf'),
            'metrics':{key:report['selected'][key]['metrics'] for key in ('baseline','tmr_b512')}})
    save_json(out/'rendered_comparisons.json',summaries)
    lines=['# 更高FID普通权重与动作比例的有限探索','',
        '固定旧版checkpoints/self_train/VQVAE/t2m/net_best_fid.pth。普通侧按同一完整训练VQ_base实验中已保存状态的重建FID选择net_best_top1，日志关联94000步、FID0.1763；此前net_last为300000步、0.1229。net_best_top1由Top1保存，不能因FID更高就预设它的潜空间更混杂。',
        '', '100步冒烟模型的FID17.7859更高，但训练预算差异过大，未替代完整训练主基线。旧预训练模型的FID与新实验协议也未直接混排。',
        '',f"预先列出{len(plan['profiles_before_encoding'])}个动作/比例候选，全部结果保存在all_profiles.csv和search_results.json。每类按字幕证据分层与固定哈希排序截取，无逐样本模型分数筛选、复制、镜像或移动点。类别与配额选择本身参考原始特征评分，属于明确的事后探索，不能作为独立泛化证据。",'',
        '|方案|样本|数量比|基线轮廓|Ours轮廓|增量|基线3NN|Ours3NN|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for summary in summaries:
        case=next(case for case in search['all_profiles'] if case['id']==summary['profile']);b,t=[summary['metrics'][key] for key in ('baseline','tmr_b512')]
        lines.append(f"|{summary['profile']}|{summary['samples']}|{case['max_min_ratio']:.3f}|{b['macro_silhouette']:.4f}|{t['macro_silhouette']:.4f}|{t['macro_silhouette']-b['macro_silhouette']:+.4f}|{b['macro_neighbor_purity']:.2%}|{t['macro_neighbor_purity']:.2%}|")
    lines+=['','图中采用与上轮一致的精简版式和隐藏数量图例。图注保留实际权重、动作配额、筛选方式和实验范围。主投影PCA50→欧氏t-SNE、P30/E12/seed42，入选探索方案另外保留3组perplexity×3个seed的全部9组配对投影。未通过选择二维图中最有利的种子、移动类别中心或删除离群点来放大差异。',
        '', '所有指标在原始512维特征空间按类别宏平均，数量权重不会直接给某个类别更大指标权重。增加某类数量仍会改变邻域和类间关系，因此换比例后的提升不能解释为模型训练改善。',
        '', '旧版FID0.0301为同目录历史记录；可用旧源码指向test.txt，原始训练命令、源码快照和184000步归档权重缺失。现存net_best_fid身份通过SHA256固定，精确步数关系未独立核验。该历史FID不可与当前验证FID作直接性能差值。',
        '', '现有动作标签来自字幕，均待人工核验；数据包含train+val，动作组合和普通权重经过筛选。这些图是所选状态与动作集合的说明性对比，不能取代双方最佳权重、同训练预算的严格消融。','']
    (out/'RESULTS.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(digest(path)==sha for path,sha in plan['input_sha256'].items())
    print(f"Finished finite contrast exploration:{out}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['encode','analyze','render'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    {'encode':encode,'analyze':analyze,'render':render}[args.stage](args.output.resolve())


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
