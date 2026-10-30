#!/usr/bin/env python3
"""About 1,200 independent motion clips, accompanied by original-space separation plots."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from threadpoolctl import threadpool_limits

from analyze import latent_metrics, run_tsne
from common import PROJECT, digest, read_rows, save_json, write_rows
from optimize_display import select_indices, score, fit_cosine
from render_polished import FAMILIES, PALETTE, render, style_panel


SETTINGS = ((30, 42), (30, 0), (50, 42), (50, 0))


def silhouette_panel(ax, labels, values, classes, metric):
    # Equal-height class bands match the macro average; no class is hidden.
    for i, label in enumerate(classes):
        points = np.sort(values[labels == label])
        y = np.linspace(i+.12, i+.88, len(points))
        ax.fill_betweenx(y, 0, points, color=PALETTE[label], alpha=.78, linewidth=0)
        ax.plot(points, y, color=PALETTE[label], lw=.65)
    ax.set(ylim=(len(classes), 0), xlim=(-1, 1),
           yticks=np.arange(len(classes))+.5, yticklabels=classes)
    ax.axvline(0, color="#9BA6B2", lw=.8)
    ax.axvline(metric, color="#31465C", lw=1, linestyle=(0, (4, 3)))
    ax.grid(axis="x", color="#E5E9EE", lw=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10, colors="#4D5B6A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Cosine silhouette in the original 512-D space", fontsize=10, color="#536273")


def render_separation(out, rows, selected, values, coordinates=None):
    labels = np.array([r["label"] for r in rows])
    classes = list(dict.fromkeys(labels))
    combined = coordinates is not None
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(15.8, 11 if combined else 6.4), facecolor="white")
    for i, family in enumerate(FAMILIES):
        x0 = .075 + i*.48
        metrics = selected[family]["metrics"]
        if combined:
            ax = fig.add_axes((x0, .575, .415, .34))
            style_panel(ax, coordinates[family], labels, classes,
                        np.random.default_rng(42).permutation(len(rows)), marker_size=18)
            ax.set_title("(a) Baseline VQ-VAE" if i == 0 else "(b) Ours: TMR-guided VQ-VAE",
                         fontsize=14, weight="bold", pad=14)
            ax = fig.add_axes((x0, .12, .415, .34))
            title = "(c) Baseline: separation by action" if i == 0 else "(d) TMR-guided: separation by action"
        else:
            ax = fig.add_axes((x0, .23, .415, .57))
            title = "(a) Baseline VQ-VAE" if i == 0 else "(b) Ours: TMR-guided VQ-VAE"
        silhouette_panel(ax, labels, values[family], classes, metrics["macro_silhouette"])
        ax.set_title(title, fontsize=12.6, weight="bold", pad=15, loc="left")
        fig.text(x0+.2075, .061 if combined else .12,
                 f"Macro silhouette: {metrics['macro_silhouette']:.4f}   |   3-NN purity: {metrics['macro_neighbor_purity']:.2%}",
                 ha="center", fontsize=10.7, color="#32495D")
    fig.text(.075, .97 if combined else .94,
             f"Same {len(rows):,} real motion clips | 10 actions | HumanML3D train + validation",
             fontsize=13, weight="bold", color="#293C50")
    if combined:
        fig.text(.075, .505, "Original-space evidence: positive silhouette means closer to its own action than to the nearest other action.",
                 fontsize=10, color="#677687")
    else:
        fig.text(.075, .875, "Positive values indicate better separation; dashed lines show the macro average. Equal-height action bands.",
                 fontsize=10, color="#677687")
    fig.text(.075, .018, "One clip per original recording; caption labels pending review. Descriptive comparison with fixed existing checkpoints.",
             fontsize=8.5, color="#7B8794")
    stem = "comparison_with_evidence" if combined else "original_space_separation"
    fig.savefig(out / f"{stem}_preview.png", dpi=150)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PROJECT / "artifacts/dense_train_validation_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/comparison_1200_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    if (out / "report.json").exists():
        raise FileExistsError("Preserving the completed comparison")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((parent / "config_snapshot.json").read_text())
    prior = json.loads((parent / "report.json").read_text())
    audit = json.loads((parent / "selection_audit.json").read_text())
    all_rows = read_rows(parent / "samples.csv")
    if digest(parent / "samples.csv") != prior["cohort_sha256"]:
        raise ValueError("Stale source manifest")
    classes = [c["name"] for c in cfg["classes"]]
    indices = select_indices(all_rows, classes, 450, 20261029)
    rows = [all_rows[i] for i in indices]
    ids = np.array([r["sample_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    if len(set(sources)) != len(rows) or len(set(ids)) != len(rows):
        raise ValueError("Repeated source or clip")
    inputs = {str(parent / name): digest(parent / name) for name in (
        "samples.csv", "report.json", "selection_audit.json", "baseline_features.npz", "tmr_b512_features.npz")}
    plan = {"purpose": "About 1200 real clips and evidence of original-space separation",
            "post_hoc_descriptive_design": True, "source_input_sha256": inputs,
            "selection": "All ten actions, cap450, seed20261029 SHA256(sample_id), no feature-based filtering",
            "class_counts": dict(Counter(labels)), "samples": len(rows),
            "primary": "PCA50 full SVD -> Euclidean t-SNE, P30 seed42, PCA initialization, 2000 iterations, auto learning rate",
            "rationale": "Restore the original comparison projection as primary; retain the newer cosine projection as a supplement",
            "stability": [list(v) for v in SETTINGS],
            "supplement": "Cosine t-SNE on original512D, P30 seed42",
            "checkpoints": {f: prior["selected"][f]["sha256"] for f in FAMILIES},
            "new_training": False, "teacher_features_used": False, "independent_test": False}
    protocol_path = out / "protocol_before_projection.json"
    if protocol_path.exists() and (not args.resume or json.loads(protocol_path.read_text()) != plan):
        raise ValueError("Use --resume with the unchanged protocol")
    save_json(protocol_path, plan)
    write_rows(out / "samples.csv", rows)
    selected, coordinates, cosine, silhouettes, fits = {}, {}, {}, {}, []
    print(f"Locked {len(rows)} independent clips; {dict(Counter(labels))}", flush=True)
    for family in FAMILIES:
        with np.load(parent / f"{family}_features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], [r["sample_id"] for r in all_rows])
            if str(data["checkpoint_sha256"]) != plan["checkpoints"][family]:
                raise ValueError("Stale checkpoint provenance")
            features = data["post_mean"][indices].copy()
        np.savez_compressed(out / f"{family}_features.npz", post_mean=features, labels=labels,
                            sample_ids=ids, checkpoint_sha256=plan["checkpoints"][family])
        metrics = score(features, labels, sources)
        silhouettes[family] = latent_metrics(features, labels, sources, 3)["silhouette_per_sample"]
        selected[family] = {**prior["selected"][family], "metrics": metrics}
        pca = PCA(n_components=50, svd_solver="full")
        reduced = pca.fit_transform(features)
        np.savez_compressed(out / f"{family}_pca.npz", components=pca.components_, mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_)
        for p, seed in SETTINGS:
            path = out / f"{family}_pca_euclidean_p{p}_s{seed}.npz"
            if args.resume and path.exists():
                with np.load(path, allow_pickle=False) as stored:
                    np.testing.assert_array_equal(stored["sample_ids"], ids)
                    if str(stored["features_sha256"]) != digest(out / f"{family}_features.npz"):
                        raise ValueError("Stale coordinate cache")
                    xy, kl = stored["coordinates"].copy(), float(stored["kl_divergence"])
            else:
                xy, kl = run_tsne(reduced, cfg, p, seed)
                np.savez_compressed(path, coordinates=xy, labels=labels, sample_ids=ids,
                                    kl_divergence=kl, features_sha256=digest(out / f"{family}_features.npz"))
            fits.append({"family": family, "method": "PCA50_Euclidean", "perplexity": p, "seed": seed,
                         "kl_divergence": kl, "trustworthiness_original_cosine_k10": float(trustworthiness(features, xy, metric="cosine", n_neighbors=10))})
            if (p, seed) == (30, 42):
                coordinates[family] = xy
            save_json(out / "fits.json", fits)
            print(f"{family}: PCA50 Euclidean P{p} seed{seed}, KL={kl:.4f}", flush=True)
        xy, kl, trust = fit_cosine(features, 30, 42)
        cosine[family] = xy
        np.savez_compressed(out / f"{family}_cosine_p30_s42.npz", coordinates=xy, sample_ids=ids,
                            labels=labels, kl_divergence=kl, features_sha256=digest(out / f"{family}_features.npz"))
        fits.append({"family": family, "method": "cosine512", "perplexity": 30, "seed": 42,
                     "kl_divergence": kl, "trustworthiness_original_cosine_k10": trust})
        print(f"{family}: macro silhouette={metrics['macro_silhouette']:.4f}, 3-NN={metrics['macro_neighbor_purity']:.2%}", flush=True)
    report = {"samples": len(rows), "classes": len(classes), "class_counts": dict(Counter(labels)),
              "split": "train+val", "split_counts": dict(Counter(audit["split_by_sample_id"][ident] for ident in ids)),
              "selected": selected, "post_hoc_focus": False, "labels_reviewed": False,
              "independent_test": False, "source_recordings": len(set(sources)),
              "display": {"scope_label": "10-ACTION COMPARISON | Same 1,198 real clips on both sides",
                          "data_label": "HumanML3D train + validation", "marker_size": 19,
                          "note": "One clip per original recording; caption labels await review; fixed existing checkpoints.",
                          "embedding_label": "Post-quantization mean | PCA50 + t-SNE | P = 30, seed = 42"}}
    for annotated in (False, True):
        rendered = render(out, classes, rows, report, coordinates, annotated=annotated)
        if rendered != {f: len(rows) for f in FAMILIES}:
            raise AssertionError("Missing plotted samples")
    supplement = out / "cosine_supplement"
    supplement.mkdir(exist_ok=True)
    cosine_report = {**report, "display": {**report["display"], "embedding_label": "Cosine t-SNE on original512D | P = 30, seed = 42"}}
    render(supplement, classes, rows, cosine_report, cosine, annotated=True)
    render_separation(out, rows, selected, silhouettes)
    render_separation(out, rows, selected, silhouettes, coordinates)
    np.savez_compressed(out / "original_space_per_sample.npz", sample_ids=ids, labels=labels,
                        **{f"{f}_silhouette": silhouettes[f] for f in FAMILIES})
    save_json(out / "fits.json", fits)
    save_json(out / "report.json", report)
    a, b = (selected[f]["metrics"] for f in FAMILIES)
    text = f"""# 约1200点的真实模型对比

固定现有两组权重与量化后512维特征，保留全部10类。限制walk为450段，其余动作保留全部，共{len(rows)}段、{len(set(sources))}份不同原始录制。两侧样本ID、裁剪和特征定义一致。选样按固定哈希顺序，不按结果筛选。

主图恢复原始协议PCA50→欧氏t-SNE，固定P30、seed42；额外保留P30/50×seed0/42及原512维余弦t-SNE补充。未按图形挑主图种子，也未修改模型或移动点来形成类别簇。

|原始特征指标|普通VQ|TMR VQ|
|---|---:|---:|
|宏平均余弦Silhouette|{a['macro_silhouette']:.4f}|{b['macro_silhouette']:.4f}|
|宏平均3-NN同类比例|{a['macro_neighbor_purity']:.2%}|{b['macro_neighbor_purity']:.2%}|

`comparison_polished`是双面板主图；`comparison_with_evidence`在散点图下面增加逐动作的原空间silhouette分布；`original_space_separation`为单独的证据图。分布图每类等高以对应宏平均，保留所有正负样本值。它不是新的降维散点，不应解读为t-SNE坐标。

二维图中的紧凑外观不足以判定原潜空间优劣。现有TMR模型并未在所有动作上同时提高紧凑性与分离性；这轮展示应描述这组数据实际观测的差异。

数据包含训练集与验证集，标签待人工核验，属于描述性探索，未进行独立测试或新训练。PNG主图600dpi，证据组合图400dpi，另提供矢量PDF/SVG。
"""
    (out / "RESULTS.zh-CN.md").write_text(text, encoding="utf-8")
    if inputs != {path: digest(path) for path in inputs}:
        raise AssertionError("An original input changed")
    save_json(out / "verification.json", {"status": "complete", "samples_per_panel": len(rows),
        "source_recordings": len(set(sources)), "same_samples_both_models": True,
        "features_unchanged_from_parent": True, "teacher_features_used": False, "new_training": False,
        "source_sha256": inputs, "renderer_sha256": digest(Path(__file__)),
        "output_sha256": {str(path.relative_to(out)): digest(path) for path in out.rglob("*") if path.suffix in (".npz", ".png", ".pdf", ".svg")}})
    print(f"Completed comparison: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
