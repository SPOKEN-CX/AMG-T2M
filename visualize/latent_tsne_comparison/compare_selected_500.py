#!/usr/bin/env python3
"""Use the requested GPT-linked TMR-VQ with a weaker saved baseline on fixed 500 clips."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, read_rows, save_json, write_rows
from compare_gpt_linked_vq import CLASSES, FAMILIES, PRIMARY, SETTINGS, cohort, filename
from optimize_display import fit_cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/selected_checkpoints_500_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; preserve earlier figures")
    parent = PROJECT / "artifacts/gpt120k_linked_vq_500_v1"
    weak_cache = PROJECT / "artifacts/checkpoint_contrast_v1"
    rows = read_rows(parent / "samples.csv")
    prior = json.loads((parent / "report.json").read_text())
    earlier = json.loads((weak_cache / "report.json").read_text())
    validation_path = PROJECT / "artifacts/expanded_validation_v1/checkpoint_probe/report.json"
    validation = json.loads(validation_path.read_text())
    audit_path = PROJECT / "artifacts/dense_train_validation_v1/selection_audit.json"
    audit = json.loads(audit_path.read_text())
    identity_path = REPO / "output/hml3d/GPT_amg_b512_full_bf16/vq_identity.json"
    identity = json.loads(identity_path.read_text())
    log_path = REPO / "output/hml3d/GPT_amg_b512_full_bf16/run.log"
    matching = [line for line in log_path.read_text().splitlines()
                if "2026-10-29 20:42:20,715" in line and "Eva. Iter 120000" in line and "FID. 0.3372" in line]
    assert len(matching) == 1
    weak = earlier["models"]["baseline_weaker"]
    assert weak["name"] == "net_last"
    for state in (weak, prior["selected"]["tmr_b512"]):
        assert digest(state["path"]) == state["sha256"]
    assert prior["selected"]["tmr_b512"]["sha256"] == identity["vq_sha256"]
    inputs = {str(path): digest(path) for path in (
        parent / "samples.csv", parent / "report.json", parent / "tmr_b512_features.npz",
        weak_cache / "samples.csv", weak_cache / "report.json", weak_cache / "baseline_weaker_features.npz",
        validation_path, audit_path, identity_path)}
    states = {
        "baseline": {"name": weak["name"], "path": weak["path"], "sha256": weak["sha256"],
                     "vq_iteration": 300000, "logged_validation_fid": .1229},
        "tmr_b512": {key: value for key, value in prior["selected"]["tmr_b512"].items() if key != "metrics"},
    }
    plan = {
        "user_requested_evaluation": matching[0], "evaluation_log": str(log_path),
        "tmr_selection": "Actual frozen VQ loaded by the user-specified GPT120000 evaluation (generation FID0.3372)",
        "vq_identity": identity,
        "baseline_selection": "User-requested weaker saved VQ: net_last; weaker earlier108-clip semantic validation and worse logged reconstruction FID than net_best_fid",
        "earlier_baseline_candidates": [{"name": s["name"], "sha256": s["sha256"],
            "macro_silhouette": s["metrics"]["macro_silhouette"],
            "macro_3nn_purity": s["metrics"]["macro_neighbor_purity"]}
            for s in validation["all_candidates"]["baseline"]],
        "states": states, "samples": 500, "class_counts": prior["class_counts"],
        "only_changed_model_from_previous500": "baseline: net_best_fid -> net_last",
        "same_full_manifest_and_crops": True, "same_feature_definition": True,
        "primary_projection": {"pca": 50, "svd_solver": "full", "setting": list(PRIMARY), "max_iter": 3000},
        "all_settings": [list(v) for v in SETTINGS],
        "input_sha256": inputs, "checkpoint_selection_uses_new500_scores": False,
        "new_training": False, "new_motion_data": False, "new_inference": False,
        "scope": "Explicit selected-checkpoint illustration; not best-versus-best or matched causal ablation",
        "split": "train+val", "labels_reviewed": False, "independent_test": False,
        "previous_best_baseline_reference": str(parent / "journal_style_v1/journal_comparison.pdf"),
    }
    save_json(out / "protocol_before_projection.json", plan)
    (out / "requested_evaluation_line.txt").write_text(matching[0]+"\n", encoding="utf-8")
    ids = np.array([r["sample_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    cache_rows = read_rows(weak_cache / "samples.csv")
    cache_ids = np.array([r["sample_id"] for r in cache_rows])
    by_id = {row["sample_id"]: i for i, row in enumerate(cache_rows)}
    indices = np.array([by_id[ident] for ident in ids])
    assert all(row == cache_rows[i] for row, i in zip(rows, indices))
    assert len(rows) == len(set(ids)) == len(set(sources)) == 500
    with np.load(weak_cache / "baseline_weaker_features.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sample_ids"], cache_ids)
        assert str(data["checkpoint_sha256"]) == weak["sha256"]
        baseline = data["post_mean"][indices].copy()
        np.testing.assert_array_equal(data["labels"][indices], labels)
    with np.load(parent / "tmr_b512_features.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sample_ids"], ids)
        np.testing.assert_array_equal(data["labels"], labels)
        assert str(data["checkpoint_sha256"]) == identity["vq_sha256"]
        tmr = data["post_mean"].copy()
    vectors = {"baseline": baseline, "tmr_b512": tmr}
    for x in vectors.values():
        assert x.shape == (500, 512) and np.isfinite(x).all()
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    report = cohort(out, rows, sources, vectors, states, SETTINGS)
    # With the same data and deterministic settings, the TMR projection must be unchanged.
    for setting in SETTINGS:
        with np.load(parent / filename("tmr_b512", setting), allow_pickle=False) as before, np.load(out / filename("tmr_b512", setting), allow_pickle=False) as after:
            np.testing.assert_array_equal(before["coordinates"], after["coordinates"])
    design = report["journal_design"]
    design.update({
        "context": "Selected checkpoint contrast  |  500 paired clips  |  6 actions",
        "subtitles": {
            "baseline": "net_last | Lower-scoring saved VQ checkpoint | step 300,000",
            "tmr_b512": "GPT-paired frozen TMR-VQ | GPT 120k generation FID 0.3372"},
        "footer_bottom": "Selected checkpoint states. GPT FID measures generation; latent metrics use the paired frozen VQ.",
        "scope": plan["scope"],
        "scope_note": "\n普通侧为用户指定的较弱net_last，TMR侧锁定2026-10-29 20:42:20的GPT120000步评估所实际配套的冻结VQ。FID0.3372是GPT生成指标，VQ参数身份由vq_identity.json及文件哈希核验。\n"})
    report["selection_protocol"] = plan
    report["split_counts"] = prior["split_counts"]
    save_json(out / "report.json", report)
    alternative = out / "cosine_supplement"
    alternative.mkdir()
    write_rows(alternative / "samples.csv", rows)
    alternative_report = copy.deepcopy(report)
    alternative_report["journal_design"]["coordinate_files"] = {}
    alternative_report["journal_design"]["footer_right"] = "Cosine t-SNE on 512-D; P=30, seed=42."
    for family in FAMILIES:
        if family == "tmr_b512":
            with np.load(parent / "cosine_supplement/tmr_b512_cosine_p30_s42.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["sample_ids"], ids)
                xy, kl = data["coordinates"].copy(), float(data["kl_divergence"])
                trust = float(data["trustworthiness_original_cosine_k10"])
        else:
            xy, kl, trust = fit_cosine(vectors[family], 30, 42)
        path = alternative / f"{family}_cosine_p30_s42.npz"
        np.savez_compressed(path, coordinates=xy, sample_ids=ids, labels=labels, kl_divergence=kl,
                            trustworthiness_original_cosine_k10=trust,
                            features_sha256=digest(out / f"{family}_features.npz"))
        alternative_report["journal_design"]["coordinate_files"][family] = path.name
    save_json(alternative / "report.json", alternative_report)
    assert digest(out / "samples.csv") == digest(parent / "samples.csv")
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save_json(out / "verification.json", {
        "same_500_manifest_byte_for_byte": True, "same_windows_and_original_recordings": True,
        "baseline_weak_cache_exact_rows_and_weights_verified": True,
        "tmr_hash_matches_user_requested_gpt_vq_identity": True,
        "tmr_features_and_all9_pca_tsne_projections_unchanged": True,
        "all_features_finite_and_unit_norm": True, "input_artifacts_unchanged": True,
        "changed_only_baseline_checkpoint": True,
    })
    b, t = [report["selected"][f]["metrics"] for f in FAMILIES]
    old_b = prior["selected"]["baseline"]["metrics"]
    text = f"""# 指定TMR模型与较弱普通VQ的500段对比

本轮只把普通侧从net_best_fid改为net_last，不改TMR模型、500段样本、裁剪、特征、主参数或版式。

TMR侧按用户最新明确指定，锁定2026-10-29 20:42:20、GPT120000步、生成FID0.3372对应的模型组合；使用该次评估加载的冻结VQ，文件SHA256为{identity['vq_sha256']}。这不是120000步VQ训练权重。其来源为batch512训练的16000步best-FID VQ，VQ验证FID约0.06644。

普通侧net_last为300000步保存权重，训练日志FID0.1229。此前108段开发诊断中其轮廓系数约0.00724，低于best-FID权重的0.04631；选择不依赖当前500段的新分数。它是较弱已保存状态，不宣称为所有训练过程中最差状态。

|当前500段原始512维宏平均指标|普通net_last|指定TMR-VQ|变化|
|---|---:|---:|---:|
|余弦轮廓系数|{b['macro_silhouette']:.4f}|{t['macro_silhouette']:.4f}|{t['macro_silhouette']-b['macro_silhouette']:+.4f}|
|3-NN同类比例|{b['macro_neighbor_purity']:.2%}|{t['macro_neighbor_purity']:.2%}|{100*(t['macro_neighbor_purity']-b['macro_neighbor_purity']):+.2f}个百分点|

上一版普通best-FID在同一500段上为{old_b['macro_silhouette']:.4f}与{old_b['macro_neighbor_purity']:.2%}，原图保留在{parent}。

六类数量不变：run117、jump172、sit47、kick57、throw70、clap37。量化后均值512维L2归一化；主图PCA50→欧氏t-SNE，P30/E12/seed42，最多3000次迭代；全部9组配对参数图、原空间相似度与逐类增量、固定参数cosine补充图均保留。TMR全部九组投影与上一版逐点完全一致。

这是明确选取较弱普通权重的所选状态对比，不能写成双方最佳性能差距或严格TMR因果消融。数据仍为训练加验证的探索子集，标签待核验。没有新增数据、训练、教师输出或按新结果筛选样本。
"""
    (out / "RESULTS.zh-CN.md").write_text(text, encoding="utf-8")
    print(f"Completed requested selected-checkpoint contrast: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
