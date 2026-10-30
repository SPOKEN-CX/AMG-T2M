#!/usr/bin/env python3
"""Create an explicitly post-hoc action-subset view and preserve the full comparison."""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from analyze import latent_metrics
from common import PROJECT, digest, read_rows, save_json, write_rows
from compare_gpt_linked_vq import cohort


ROOT = PROJECT / "artifacts/legacy_contrast_exploration_v1"
SOURCE = ROOT / "recommended_mature_comparison"
OUT = ROOT / "overlap_review_v1"
FAMILIES = ("baseline", "tmr_b512")
SETTINGS = [(20,12,42), (30,12,42), (50,12,42)]


def main():
    target = OUT / "five_class_602"
    if target.exists():
        raise FileExistsError("Preserve completed views; choose a new output directory")
    report = json.loads((SOURCE / "report.json").read_text())
    audit = json.loads((SOURCE / "selection_audit.json").read_text())
    rows = read_rows(SOURCE / "samples.csv")
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    assert len(rows) == len(set(ids)) == len(set(sources)) == 700
    assert dict(Counter(labels)) == report["class_counts"]
    vectors, diagnostics = {}, {}
    for family in FAMILIES:
        with np.load(SOURCE / f"{family}_features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            assert str(data["checkpoint_sha256"]) == report["selected"][family]["sha256"]
            vectors[family] = data["post_mean"].copy()
        metric = latent_metrics(vectors[family], labels, sources, 3)
        diagnostics[family] = {}
        for label in report["class_counts"]:
            mask = labels == label
            near = labels[metric["neighbors"][mask].reshape(-1)]
            others = Counter(near[near != label].tolist())
            diagnostics[family][label] = {
                "samples": int(mask.sum()),
                "silhouette": float(metric["silhouette_per_sample"][mask].mean()),
                "three_nn_purity": float(metric["knn_purity_per_sample"][mask].mean()),
                "other_class_neighbors": dict(others.most_common()),
            }
    jointly_lowest = min(report["class_counts"], key=lambda label: np.mean([
        diagnostics[f][label]["three_nn_purity"] for f in FAMILIES]))
    assert jointly_lowest == "throw"
    keep = labels != jointly_lowest
    clips = [row for row, chosen in zip(rows, keep) if chosen]
    excluded = [row for row, chosen in zip(rows, keep) if not chosen]
    selected_sources = sources[keep]
    classes = [label for label in report["class_counts"] if label != jointly_lowest]
    assert len(clips) == 602 and len(excluded) == 98
    states = {family: {key: value for key,value in report["selected"][family].items() if key != "metrics"}
              for family in FAMILIES}
    paths = [SOURCE / "samples.csv", SOURCE / "selection_audit.json", SOURCE / "report.json"]
    paths += [SOURCE / f"{family}_features.npz" for family in FAMILIES]
    plan = {
        "source": str(SOURCE), "source_hashes": {str(path): digest(path) for path in paths},
        "source_samples": 700, "remaining_samples": 602, "excluded_samples": 98,
        "removed_action": jointly_lowest,
        "selection_rule": "Lowest mean 3-NN same-class purity across both models on this same selected cohort",
        "diagnostics": diagnostics,
        "selection_scope": "Post-hoc action-level exploration using the same cohort; label correctness is not established",
        "individual_samples_filtered_by_model_scores": False,
        "same_sample_ids_and_windows_in_both_panels": True,
        "class_labels_pending_human_review": True,
        "settings": SETTINGS, "primary": [30,12,42],
        "feature": "Original L2-normalized512-D post-quantization mean; no re-encoding",
        "coordinates_refitted_after_action_removal": True,
        "full_six_class_view_retained": str(OUT / "full_700"),
    }
    save_json(OUT / "selection_plan.json", plan)
    write_rows(OUT / "excluded_throw_98.csv", excluded)
    selected = {family: vectors[family][keep] for family in FAMILIES}
    new = cohort(target, clips, selected_sources, selected, states, SETTINGS, classes=classes)
    new.pop("sample_selection_uses_model_features", None)
    new["individual_sample_ranking_uses_model_features"] = False
    new["action_and_quota_selection_uses_model_metrics"] = True
    new["selection_protocol"] = plan
    new["journal_design"].update({
        "titles": ["(a) Baseline VQ-VAE", "(b) Ours (w/ Contrastive Learning)"],
        "context": "Post-hoc action subset | 602 paired clips",
        "scope": "Post-hoc removal of one overlapping action, with the full six-class control retained",
        "scope_note": "\n基于同一700段中两模型原始512维3-NN同类比例，投掷类别两侧均存在明显跨类邻居，与挥手交叠较多；因此探索性整类移除98段。类别选择发生在看过本数据结果后，不能据此断定标签错误或模型在全体动作上更优。两侧使用同样602段重新拟合t-SNE；完整700段重绘版和被移除清单保留。\n",
    })
    save_json(target / "report.json", new)
    save_json(target / "selection_audit.json", {
        "counts": dict(Counter(row["label"] for row in clips)),
        "source_path_by_sample_id": {ident:audit["source_path_by_sample_id"][ident] for ident in ids[keep]},
        "split_by_sample_id": {ident:audit["split_by_sample_id"][ident] for ident in ids[keep]},
        "tier_by_sample_id": {ident:audit["tier_by_sample_id"][ident] for ident in ids[keep]},
        "labels_reviewed": False,
    })
    render = target / "compact_plot"
    subprocess.run([sys.executable, str(PROJECT / "render_clean_reference.py"), "--study", str(target),
        "--right-title", "(b) Ours (w/ Contrastive Learning)", "--hide-counts", "--compact-panels",
        "--point-size", "11", "--output", str(render)], check=True)
    caption_path = render / "FIGURE_CAPTION.txt"
    caption_path.write_text(caption_path.read_text(encoding="utf-8").strip()+
        " The throw class (98 clips) was excluded from both panels after inspecting the same cohort's "
        "nearest-neighbor diagnostics; this action-level selection is post hoc. The full six-class comparison is retained.\n",
        encoding="utf-8")
    manifest_path = render / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][caption_path.name] = digest(caption_path)
    save_json(manifest_path, manifest)
    assert all(digest(path) == sha for path, sha in plan["source_hashes"].items())
    b, t = [new["selected"][family]["metrics"] for family in FAMILIES]
    source_b, source_t = [report["selected"][family]["metrics"] for family in FAMILIES]
    lines = ["# 动作交叠检查与精简图", "",
        "原700段六动作中，投掷在普通模型与Ours的原始512维3-NN同类比例分别为"
        f"{diagnostics['baseline']['throw']['three_nn_purity']:.2%}和{diagnostics['tmr_b512']['throw']['three_nn_purity']:.2%}，"
        "两模型均与挥手有较多跨类邻居。两模型同类比例平均后，投掷在六类中最低。", "",
        "按上述同一批数据的事后诊断，整类排除98条投掷，其余602条运动两边完全相同，"
        "所有保留样本、真实特征及标签均未修改；在保留的特征上，两侧各自使用相同参数重新拟合t-SNE。"
        "交叠不是标签错误的证明；原六类700段与全部排除样本的清单均保留。", "",
        "|方案|样本|普通轮廓|Ours轮廓|普通3-NN|Ours 3-NN|", "|---|---:|---:|---:|---:|---:|",
        f"|完整六类|700|{source_b['macro_silhouette']:.4f}|{source_t['macro_silhouette']:.4f}|{source_b['macro_neighbor_purity']:.2%}|{source_t['macro_neighbor_purity']:.2%}|",
        f"|移除投掷的五类|602|{b['macro_silhouette']:.4f}|{t['macro_silhouette']:.4f}|{b['macro_neighbor_purity']:.2%}|{t['macro_neighbor_purity']:.2%}|", "",
        "完整700段：full_700/comparison_clean.pdf；五类602段：five_class_602/compact_plot/comparison_clean.pdf。"
        "两版均采用更大的相同散点和更紧凑的面板，等比例显示；没有为制造分离移动簇或删除单个离群点。", "",
        "剔除动作属于看过同一数据后进行的探索，指标上升也可能来自任务变简单，不能用来推断总体性能提升。"
        "HumanML3D train+val文本标签尚待人工复核，旧版Ours权重的历史FID口径和精确迭代身份限制见各图图注。", ""]
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"removed":jointly_lowest,"source":700,"remaining":len(clips),
                      "baseline":b["macro_silhouette"],"ours":t["macro_silhouette"],
                      "baseline_3nn":b["macro_neighbor_purity"],"ours_3nn":t["macro_neighbor_purity"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
