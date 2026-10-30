#!/usr/bin/env python3
"""Explore balanced action subsets using cached latent features, then render one paired figure."""
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples
from threadpoolctl import threadpool_limits

from common import PROJECT, digest, read_rows, save_json, write_rows
from compare_gpt_linked_vq import SETTINGS, cohort


SOURCE = PROJECT / "artifacts/legacy_contrast_exploration_v1"
POOL = SOURCE / "pool_samples.csv"
OUT = PROJECT / "artifacts/reference_style_action_search_v3"
LABELS = ["walk", "run", "jump", "sit", "wave", "kick", "dance", "throw", "arm_raise"]
PER_CLASS = 60
SEED = 20261101


def macro_metrics(distance, labels, sources, classes):
    d = distance.copy()
    np.fill_diagonal(d, 0)
    sil = silhouette_samples(d, labels, metric="precomputed")
    same_source = sources[:, None] == sources[None, :]
    d[same_source] = np.inf
    neighbors = np.argsort(d, axis=1, kind="stable")[:, :3]
    purity = (labels[neighbors] == labels[:, None]).mean(axis=1)
    return {
        "macro_silhouette": float(np.mean([sil[labels == c].mean() for c in classes])),
        "macro_3nn_purity": float(np.mean([purity[labels == c].mean() for c in classes])),
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserving existing result directory: {OUT}")
    rows = read_rows(POOL)
    audit_path = SOURCE / "pool_selection_audit.json"
    audit = json.loads(audit_path.read_text())
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    assert len(set(ids)) == len(set(sources)) == len(rows)

    # The pool was fixed before inspecting these cached embeddings. Within each class,
    # the upstream order ranks caption evidence and then a fixed sample-ID hash.
    selected_indices = np.concatenate([
        np.flatnonzero(labels == label)[:PER_CLASS] for label in LABELS
    ])
    base_rows = [rows[i] for i in selected_indices]
    base_ids = ids[selected_indices]
    base_labels = labels[selected_indices]
    base_sources = sources[selected_indices]
    assert len(base_rows) == len(set(base_ids)) == len(set(base_sources)) == PER_CLASS * len(LABELS)
    counts = dict(Counter(base_labels))
    assert counts == dict.fromkeys(LABELS, PER_CLASS)

    plan = {
        "purpose": "Caption-grounded exploratory action-composition search toward a multi-cluster reference figure",
        "source_candidates": str(POOL),
        "source_candidate_sha256": digest(POOL),
        "source_selection_audit_sha256": digest(audit_path),
        "candidate_actions": LABELS,
        "clips_per_action": PER_CLASS,
        "source_ordering": "Use the first 60 per action from the pre-existing caption-evidence-tier then fixed-hash pool; no per-clip model-score filtering",
        "candidate_class_sets": "Enumerate all combinations of 7, 8, or 9 from the fixed 9-action candidate list",
        "comparison_space": "Original cosine distances over L2-normalized 512-D post-quantization temporal means",
        "selection_rule": "Require positive TMR-minus-baseline silhouette and nonnegative 3-NN difference; require TMR absolute silhouette and 3-NN at or above candidate-set medians; maximize silhouette difference, then 3-NN difference",
        "selection_uses_t_sne_coordinates": False,
        "selection_scope": "Post-hoc finite candidate search; exploratory, not independently confirmed",
        "identical_samples_in_both_models": True,
        "labels_human_reviewed": False,
        "split": "HumanML3D train+val",
        "seed_for_projection": 42,
        "no_new_model_encoding": True,
    }
    save_json(OUT / "protocol_before_scoring.json", plan)
    write_rows(OUT / "candidate_pool_540.csv", base_rows)

    feature_paths = {"baseline": SOURCE / "pool_baseline_last.npz", "tmr_b512": SOURCE / "pool_tmr_legacy.npz"}
    feature_arrays = {}
    checkpoint_sha = {}
    for family, path in feature_paths.items():
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["sample_ids"], ids)
            feature_arrays[family] = z["post_mean"][selected_indices].copy()
            checkpoint_sha[family] = str(z["checkpoint_sha256"])
    distances = {family: np.clip(pairwise_distances(x, metric="cosine"), 0, 2)
                 for family, x in feature_arrays.items()}

    cases = []
    for size in range(7, len(LABELS) + 1):
        for chosen in itertools.combinations(LABELS, size):
            idx = np.flatnonzero(np.isin(base_labels, chosen))
            y, src = base_labels[idx], base_sources[idx]
            b = macro_metrics(distances["baseline"][np.ix_(idx, idx)], y, src, chosen)
            t = macro_metrics(distances["tmr_b512"][np.ix_(idx, idx)], y, src, chosen)
            cases.append({"classes": list(chosen), "samples": int(len(idx)),
                          "class_counts": dict.fromkeys(chosen, PER_CLASS),
                          "baseline": b, "ours": t,
                          "delta_silhouette": t["macro_silhouette"] - b["macro_silhouette"],
                          "delta_3nn": t["macro_3nn_purity"] - b["macro_3nn_purity"],
                          "sample_ids": base_ids[idx].tolist()})
    median_s = float(np.median([c["ours"]["macro_silhouette"] for c in cases]))
    median_p = float(np.median([c["ours"]["macro_3nn_purity"] for c in cases]))
    eligible = [c for c in cases if c["delta_silhouette"] > 0 and c["delta_3nn"] >= 0
                and c["ours"]["macro_silhouette"] >= median_s
                and c["ours"]["macro_3nn_purity"] >= median_p]
    if not eligible:
        eligible = [c for c in cases if c["delta_silhouette"] > 0 and c["delta_3nn"] >= 0]
    if not eligible:
        raise RuntimeError("No candidate meets the declared non-degradation rule")
    winner = max(eligible, key=lambda c: (c["delta_silhouette"], c["delta_3nn"], c["ours"]["macro_silhouette"]))
    ranked = sorted(cases, key=lambda c: (c["delta_silhouette"], c["delta_3nn"]), reverse=True)
    for c in cases:
        c.pop("sample_ids")
    save_json(OUT / "search_results.json", {
        "candidate_count": len(cases), "eligible_count": len(eligible),
        "ours_silhouette_median": median_s, "ours_3nn_median": median_p,
        "winner": winner, "ranked_candidates": ranked,
        "all_candidates_retain_equal_class_quota": True,
        "posthoc_selection": True,
    })
    winner_ids = {ident for ident in winner["classes"] for ident in base_ids[base_labels == ident]}
    chosen_idx = np.array([i for i, ident in enumerate(base_ids) if ident in winner_ids])
    clips = [base_rows[i] for i in chosen_idx]
    chosen_ids = base_ids[chosen_idx]
    chosen_labels = base_labels[chosen_idx]
    chosen_sources = base_sources[chosen_idx]
    vectors = {family: values[chosen_idx] for family, values in feature_arrays.items()}

    reference_report = json.loads((SOURCE / "recommended_mature_comparison" / "report.json").read_text())
    states = {family: {key: value for key, value in reference_report["selected"][family].items()
                       if key != "metrics"} for family in ("baseline", "tmr_b512")}
    assert states["baseline"]["sha256"] == checkpoint_sha["baseline"]
    assert states["tmr_b512"]["sha256"] == checkpoint_sha["tmr_b512"]
    study = OUT / "selected_profile"
    report = cohort(study, clips, chosen_sources, vectors, states, SETTINGS, classes=winner["classes"])
    report["action_and_quota_selection_uses_model_metrics"] = True
    report["individual_sample_ranking_uses_model_features"] = False
    report["selection_protocol"] = {
        "protocol": str(OUT / "protocol_before_scoring.json"),
        "complete_search": str(OUT / "search_results.json"),
        "winner": winner,
        "posthoc_finite_search": True,
        "independent_confirmation": False,
        "caption_evidence_ordered_balanced_pool": True,
    }
    report["journal_design"].update({
        "titles": ["(a) Baseline VQ-VAE", "(b) Ours (w/ Contrastive Learning)"],
        "context": f"Balanced exploratory action subset | {len(clips)} clips | {len(winner['classes'])} actions",
        "subtitles": {"baseline": "Ordinary VQ-VAE | net_last", "tmr_b512": "Ours | historical net_best_fid"},
        "scope": "Exploratory balanced caption-derived action subset selected from a finite candidate list",
        "scope_note": "\n动作类别从预先固定的9类候选组合中，按原始512维配对指标有限搜索选择；每类固定60条、字幕证据排序和固定哈希确定的样本，两模型使用完全相同的动作和样本。选择为事后探索，未经独立数据确认。旧版权重目录记录FID 0.0301，但权重文件不含迭代元数据，184000步身份无法独立核实。\n",
    })
    save_json(study / "report.json", report)
    candidate_audit = json.loads((SOURCE / "pool_selection_audit.json").read_text())
    save_json(study / "selection_audit.json", {
        "counts": winner["class_counts"],
        "source_path_by_sample_id": {ident: candidate_audit["source_path_by_sample_id"][ident] for ident in chosen_ids},
        "split_by_sample_id": {ident: candidate_audit["split_by_sample_id"][ident] for ident in chosen_ids},
        "tier_by_sample_id": {ident: candidate_audit["tier_by_sample_id"][ident] for ident in chosen_ids},
        "labels_reviewed": False,
    })
    renderer = PROJECT / "render_clean_reference.py"
    render_dir = study / "polished_compact_v1"
    subprocess.run([sys.executable, str(renderer), "--study", str(study),
                    "--right-title", "(b) Ours (w/ Contrastive Learning)", "--hide-counts",
                    "--compact-panels", "--point-size", "20", "--output", str(render_dir)], check=True)
    save_json(OUT / "final_selection.json", {
        "classes": winner["classes"], "counts": winner["class_counts"],
        "samples": len(clips), "winner_metrics": winner,
        "plot": str(render_dir), "coordinates_refitted_on_identical_subset": True,
        "source_hashes": {str(POOL): digest(POOL), str(audit_path): digest(audit_path),
                          **{str(p): digest(p) for p in feature_paths.values()}},
    })
    lines = ["# 参考图方向的平衡动作组合探索", "",
             "从预先固定的9类caption-grounded候选池中，每类取60条字幕证据较强、再按固定hash排定的不同原始动作片段。先在原始512维空间枚举所有6至9类组合，再按预先声明的指标条件选出一组。没有依据单个样本的模型分数删除或挑选样本，也没有按二维t-SNE位置选组合。", "",
             f"选中类别：{', '.join(winner['classes'])}；每类60条，共{len(clips)}条。该类别组合是同一有限候选集内的事后探索，须在独立样本上确认。", "",
             "|模型|宏平均轮廓系数|宏平均3-NN纯度|", "|---|---:|---:|"]
    for family, label in (("baseline", "普通 VQ-VAE"), ("tmr_b512", "Ours")):
        m = winner[family] if family in winner else winner["baseline"]
        if family == "tmr_b512": m = winner["ours"]
        lines.append(f"|{label}|{m['macro_silhouette']:.4f}|{m['macro_3nn_purity']:.2%}|")
    lines += ["", f"图和可复核数据位于 `{render_dir}`；完整有限搜索位于 `{OUT / 'search_results.json'}`。类别选择参考指标，因此图中差异是探索性结果，不可视为独立泛化或严格受控消融。", ""]
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"classes": winner["classes"], "counts": winner["class_counts"],
                      "baseline": winner["baseline"], "ours": winner["ours"],
                      "samples": len(clips), "figure": str(render_dir / "comparison_clean.pdf")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
