#!/usr/bin/env python3
"""Expand the existing six-action cohort to 500, without feature-based selection."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from common import PROJECT, digest, read_rows, save_json, write_rows
from compare_gpt_linked_vq import CLASSES, FAMILIES, SEED, SETTINGS, cohort
from optimize_display import fit_cosine, score, select_indices


def balanced_quotas(available, target):
    """Distribute counts evenly until a class runs out; depends only on counts."""
    if target < 1 or target > sum(available.values()):
        raise ValueError("Target must fit the available unique clips")
    quotas = dict.fromkeys(available, 0)
    for _ in range(target):
        eligible = [label for label in available if quotas[label] < available[label]]
        label = min(eligible, key=lambda label: (quotas[label], list(available).index(label)))
        quotas[label] += 1
    return quotas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/gpt120k_linked_vq_500_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Preserve prior results; choose a new output directory")
    parent = PROJECT / "artifacts/gpt120k_linked_vq_six_actions_v1"
    source = parent / "all_588"
    rows = read_rows(source / "samples.csv")
    previous_rows = read_rows(parent / "balanced_222/samples.csv")
    previous = json.loads((parent / "balanced_222/report.json").read_text())
    prior = json.loads((source / "report.json").read_text())
    audit_path = PROJECT / "artifacts/dense_train_validation_v1/selection_audit.json"
    audit = json.loads(audit_path.read_text())
    available = {label: sum(row["label"] == label for row in rows) for label in CLASSES}
    quotas = balanced_quotas(available, 500)
    assert quotas == dict(zip(CLASSES, (117, 172, 47, 57, 70, 37)))
    indices = np.concatenate([select_indices(rows, [label], n, SEED, balanced=True)
                              for label, n in quotas.items()])
    selected_rows = [rows[i] for i in indices]
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    chosen_ids = ids[indices]
    assert len(set(chosen_ids)) == 500
    assert set(row["sample_id"] for row in previous_rows).issubset(set(chosen_ids))
    assert Counter(row["label"] for row in selected_rows) == Counter(quotas)
    sources = np.array([audit["source_path_by_sample_id"][str(ident)] for ident in chosen_ids])
    assert len(set(sources)) == 500
    states = {family: {key: value for key, value in prior["selected"][family].items() if key != "metrics"}
              for family in FAMILIES}
    inputs = {str(path): digest(path) for path in (
        source / "samples.csv", source / "report.json", source / "baseline_features.npz",
        source / "tmr_b512_features.npz", parent / "balanced_222/samples.csv",
        parent / "balanced_222/report.json", audit_path)}
    for state in states.values():
        assert digest(state["path"]) == state["sha256"]
    plan = {
        "target_total": 500, "available_counts": available, "selected_counts": quotas,
        "count_allocation": "Capped equal allocation from zero until 500, using only availability counts",
        "selection": "SHA256(seed:sample_id) ascending per class, with the original seed",
        "selection_seed": SEED, "all_previous_222_clips_retained": True,
        "new_clips_relative_to_222": 278,
        "additional_selection_does_not_access_features_or_coordinates": True,
        "main_models": states,
        "primary_projection": {"pca": 50, "solver": "full", "perplexity": 30,
                               "early_exaggeration": 12, "random_state": 42,
                               "init": "pca", "metric": "euclidean", "learning_rate": "auto", "max_iter": 3000},
        "all_sensitivity_settings": [list(setting) for setting in SETTINGS],
        "feature": "Same cached 512-D L2-normalized post-quantization temporal mean features; identical 64-frame crops",
        "split": "train+val", "labels_reviewed": False, "independent_test": False,
        "class_selection_informed_by_previous_diagnostics": True,
        "new_raw_data_or_labels_added": False, "new_inference": False, "new_training": False,
        "new_clips_are_existing_candidates": True,
        "input_sha256": inputs,
    }
    save_json(out / "protocol_before_projection.json", plan)
    vectors, all_vectors = {}, {}
    for family in FAMILIES:
        with np.load(source / f"{family}_features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            assert str(data["checkpoint_sha256"]) == states[family]["sha256"]
            all_vectors[family] = data["post_mean"].copy()
            vectors[family] = all_vectors[family][indices]
            assert np.isfinite(vectors[family]).all()
            np.testing.assert_allclose(np.linalg.norm(vectors[family], axis=1), 1, atol=1e-6)
    report = cohort(out, selected_rows, sources, vectors, states, SETTINGS)
    report["journal_design"]["context"] = "Expanded action subset  |  500 paired clips  |  6 actions"
    report["sample_selection"] = plan
    report["split_counts"] = dict(Counter(audit["split_by_sample_id"][str(ident)] for ident in chosen_ids))
    save_json(out / "report.json", report)
    alternative = out / "cosine_supplement"
    alternative.mkdir()
    write_rows(alternative / "samples.csv", selected_rows)
    alternative_report = copy.deepcopy(report)
    alternative_report["journal_design"]["coordinate_files"] = {}
    alternative_report["journal_design"]["footer_right"] = "Cosine t-SNE on 512-D; P=30, seed=42."
    alternative_report["journal_design"]["footer_bottom"] = "Original 512-D macro metrics. Alternative projection; same frozen VQ and 500 clips."
    for family in FAMILIES:
        xy, kl, trust = fit_cosine(vectors[family], 30, 42)
        path = alternative / f"{family}_cosine_p30_s42.npz"
        np.savez_compressed(path, coordinates=xy, sample_ids=chosen_ids, labels=labels[indices],
                            kl_divergence=kl, trustworthiness_original_cosine_k10=trust,
                            features_sha256=digest(out / f"{family}_features.npz"))
        alternative_report["journal_design"]["coordinate_files"][family] = path.name
    save_json(alternative / "report.json", alternative_report)
    stability = []
    for seed in range(SEED, SEED+5):
        idx = np.concatenate([select_indices(rows, [label], n, seed, balanced=True)
                              for label, n in quotas.items()])
        recording_ids = np.array([audit["source_path_by_sample_id"][str(ident)] for ident in ids[idx]])
        stability.append({"seed": seed, "metrics": {
            family: score(x[idx], labels[idx], recording_ids) for family, x in all_vectors.items()}})
    save_json(out / "sampling_stability.json", {
        "scope": "Overlapping sample repeats; only jump is subsampled, all other class candidates retained; not training or independent test repeats",
        "class_counts": quotas, "results": stability})
    comparison = []
    for name, item in (("balanced_222", previous), ("expanded_500", report), ("all_588", prior)):
        b, t = [item["selected"][family]["metrics"] for family in FAMILIES]
        comparison.append({"cohort": name, "samples": item["samples"], "baseline": b, "tmr": t,
                           "silhouette_delta": t["macro_silhouette"]-b["macro_silhouette"],
                           "purity_delta_pp": 100*(t["macro_neighbor_purity"]-b["macro_neighbor_purity"])})
    save_json(out / "cohort_comparison.json", comparison)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save_json(out / "verification.json", {
        "exactly_500_distinct_samples_and_recordings": True, "class_counts": quotas,
        "all_previous_222_retained": True, "278_additional_existing_clips": True,
        "feature_cache_ids_labels_checkpoint_hashes_match": True,
        "all_features_finite_and_unit_norm": True, "all_source_artifacts_unchanged": True,
        "same_windows_and_features_as_previous_comparison": True,
        "old_figures_and_checkpoints_preserved": True,
    })
    lines = ["# 500段六类动作对比", "",
             "从上一轮588段候选中取500段，完整保留原均衡222段，并增加278段现有运动；未下载、复制或合成数据。", "",
             "六类数量按可用上限尽量均分：run117、jump172、sit47、kick57、throw70、clap37。样本按原种子与ID哈希排序，抽样不读取模型特征或二维坐标。因少数类不足，这不是每类等量集合。", "",
             "普通模型仍为best-FID31000步权重，TMR仍为指定GPT实际配套的best-FID16000步冻结权重。特征与64帧裁剪保持一致，不进行新训练或推理。", "",
             "|集合|普通→TMR轮廓系数|增量|普通→TMR 3-NN纯度|增量（百分点）|",
             "|---|---:|---:|---:|---:|"]
    for item in comparison:
        b, t = item["baseline"], item["tmr"]
        lines.append(f"|{item['cohort']}|{b['macro_silhouette']:.4f}→{t['macro_silhouette']:.4f}|{item['silhouette_delta']:+.4f}|{b['macro_neighbor_purity']:.2%}→{t['macro_neighbor_purity']:.2%}|{item['purity_delta_pp']:+.2f}|")
    lines += ["", "主图固定PCA50→欧氏t-SNE，P30、E12、seed42、最多3000次迭代，与上一轮一致。另保存P20/30/50×seed0/1/42全部9组配对降维图，以及固定参数的原始512维cosine投影。所有图都完整保留500个真实点。", "",
              "主指标是原始512维余弦轮廓系数与3-NN同类比例，按类别宏平均。原空间相似度矩阵、逐类增量与5-NN保留在配套文件。条件重抽样文件使用固定逐样本分数的微平均区间，不能与面板宏平均混淆，也不包含训练与标签不确定性。", "",
              "222、500、588集合的类别比例不同，不能将跨集合绝对分数的变化归因于模型改善或样本量本身。数据包含训练与验证，标签待人工核验，类别参考此前诊断选取；结果仍属探索性展示。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed 500-clip expansion: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
