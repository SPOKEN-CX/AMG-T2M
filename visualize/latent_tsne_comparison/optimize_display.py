#!/usr/bin/env python3
"""Moderate, reproducible action comparisons, with the full diagnostic context retained."""
import argparse
import csv
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits

from analyze import latent_metrics
from common import PROJECT, digest, read_rows, save_json, write_rows
from render_polished import FAMILIES, PALETTE, render


PROFILES = (
    {"name": "overview_490", "classes": None, "cap": 60, "balanced": False,
     "scope_label": "10-ACTION OVERVIEW | Up to 60 clips per action",
     "rationale": "Retain all ten actions, cap dominant classes by a feature-blind deterministic ranking; retain all smaller classes"},
    {"name": "focused_200", "classes": ["run", "jump", "sit", "kick", "throw"], "cap": 40, "balanced": True,
     "scope_label": "FOCUSED ACTION SUBSET | 5 actions, 40 clips each | Exploratory",
     "rationale": "Post-hoc illustrative focus on five actions with earlier full-cohort silhouette gains; not evidence of universal superiority"},
)
FIT_SETTINGS = ((30, 42), (30, 0), (50, 42), (50, 0))


def select_indices(rows, classes, cap, seed, *, balanced=False):
    """Selection has no access to features, model scores or 2D coordinates."""
    indices = []
    for label in classes:
        candidates = [i for i, row in enumerate(rows) if row["label"] == label]
        candidates.sort(key=lambda i: hashlib.sha256(f"{seed}:{rows[i]['sample_id']}".encode()).hexdigest())
        if balanced and len(candidates) < cap:
            raise ValueError(f"Insufficient unique samples for {label}")
        indices.extend(candidates[:cap])
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate selected samples")
    return np.array(indices, dtype=np.int64)


def score(features, labels, sources):
    values = latent_metrics(features, labels, sources, 3)
    by_class = {label: {"samples": int((labels == label).sum()),
                       "silhouette": float(values["silhouette_per_sample"][labels == label].mean()),
                       "neighbor_purity": float(values["knn_purity_per_sample"][labels == label].mean())}
                for label in dict.fromkeys(labels)}
    return {"macro_silhouette": float(np.mean([v["silhouette"] for v in by_class.values()])),
            "macro_neighbor_purity": float(np.mean([v["neighbor_purity"] for v in by_class.values()])),
            "micro_silhouette": values["silhouette_cosine"],
            "micro_neighbor_purity": values["knn_purity"], "per_class": by_class}


def fit_cosine(features, perplexity, seed):
    kwargs = dict(n_components=2, metric="cosine", init="pca", perplexity=perplexity,
                  random_state=seed, learning_rate="auto", early_exaggeration=12,
                  method="barnes_hut", n_jobs=1)
    parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[parameter] = 2000
    model = TSNE(**kwargs)
    coordinates = model.fit_transform(features)
    if not np.isfinite(coordinates).all() or not np.isfinite(model.kl_divergence_):
        raise ValueError("Invalid t-SNE fit")
    return coordinates, float(model.kl_divergence_), float(trustworthiness(
        features, coordinates, n_neighbors=10, metric="cosine"))


def sensitivity(path, classes, labels, fits):
    fig, axes = plt.subplots(len(FIT_SETTINGS), 2, figsize=(10.5, 11))
    for row, setting in enumerate(FIT_SETTINGS):
        for column, family in enumerate(FAMILIES):
            ax = axes[row, column]
            xy = fits[family][setting]
            for label in classes:
                selected = labels == label
                ax.scatter(xy[selected, 0], xy[selected, 1], s=10, color=PALETTE[label], alpha=.82, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_color("#D7DDE4")
            role = "Baseline" if column == 0 else "TMR-VQ-VAE"
            ax.set_title(f"{role} | P={setting[0]}, seed={setting[1]}", fontsize=10)
    fig.suptitle("Cosine t-SNE stability | identical settings and samples on both sides", fontsize=12)
    fig.tight_layout(rect=(0, .015, 1, .965))
    fig.savefig(path / "sensitivity.png", dpi=180)
    fig.savefig(path / "sensitivity.pdf")
    plt.close(fig)


def action_differences(report, classes, out):
    """All ten actions in the original 2,050-clip reference, including degradations."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    baseline = report["selected"]["baseline"]["metrics"]["per_class"]
    ours = report["selected"]["tmr_b512"]["metrics"]["per_class"]
    y = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7))
    fig.subplots_adjust(left=.09, right=.965, top=.79, bottom=.18, wspace=.32)
    for ax, key, scale, title in zip(axes, ("silhouette", "neighbor_purity"), (1, 100),
                                    ("(a) Change in cosine silhouette", "(b) Change in 3-NN purity (pp)")):
        delta = np.array([(ours[label][key]-baseline[label][key])*scale for label in classes])
        colors = ["#178F83" if value >= 0 else "#CB746B" for value in delta]
        ax.barh(y, delta, color=colors, height=.60, zorder=3)
        ax.axvline(0, color="#6B7785", lw=.85)
        ax.set_yticks(y, classes)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11.6, weight="bold", pad=15)
        ax.grid(axis="x", color="#E7EBEF", lw=.65, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        width = max(float(np.ptp(delta)), .01)
        pad = width*.035
        ax.set_xlim(float(delta.min())-width*.24, float(delta.max())+width*.25)
        ax.tick_params(axis="both", length=0, colors="#556171")
        for row, value in enumerate(delta):
            label = f"{value:+.3f}" if scale == 1 else f"{value:+.2f}"
            ax.text(value + (pad if value >= 0 else -pad), row, label, va="center",
                    ha="left" if value >= 0 else "right", fontsize=9.8, color="#354354")
    fig.text(.09, .94, "Where TMR-VQ-VAE improves", fontsize=19, weight="bold", color="#243445")
    fig.text(.09, .875, "All 10 actions | original 512-D feature space | same 2,050-clip reference cohort",
             fontsize=10.8, color="#667486")
    fig.text(.09, .085, "Positive = TMR-VQ-VAE higher; negative = baseline higher. Values are descriptive.",
             fontsize=9.2, color="#667486")
    fig.text(.09, .04, "HumanML3D train + validation | caption labels pending review | fixed existing checkpoints",
             fontsize=8.6, color="#7C8793")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"action_differences.{suffix}", dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PROJECT / "artifacts/dense_train_validation_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/moderate_comparison_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    if (out / "report.json").exists():
        raise FileExistsError("Completed comparison preserved; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(parent / "samples.csv")
    prior = json.loads((parent / "report.json").read_text())
    audit = json.loads((parent / "selection_audit.json").read_text())
    cfg = json.loads((parent / "config_snapshot.json").read_text())
    classes = [c["name"] for c in cfg["classes"]]
    if digest(parent / "samples.csv") != prior["cohort_sha256"]:
        raise ValueError("Stale source cohort")
    inputs = {str(parent / name): digest(parent / name) for name in (
        "samples.csv", "report.json", "selection_audit.json", "baseline_features.npz", "tmr_b512_features.npz")}
    plan = {"post_hoc_exploratory_design": True, "parent_input_sha256": inputs,
            "selection_seed": 20261029, "profiles": list(PROFILES),
            "embedding": {"distance": "cosine", "dimensions": 512, "pre_pca": False,
                          "initialization": "pca", "iterations": 2000, "early_exaggeration": 12,
                          "learning_rate": "auto", "primary_perplexity": 30, "primary_seed": 42,
                          "all_settings": [list(setting) for setting in FIT_SETTINGS]},
            "rationale": "Use the cosine metric already used by the original feature-space diagnostics; cap class counts without feature filtering",
            "inference_scope": "Descriptive training-inclusive visualization, pending labels; not independent confirmation or universal superiority",
            "checkpoints": {family: prior["selected"][family]["sha256"] for family in FAMILIES}}
    plan_path = out / "protocol_before_projection.json"
    if plan_path.exists():
        if not args.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Use --resume with the identical frozen protocol")
    save_json(plan_path, plan)
    vectors = {}
    ids = np.array([r["sample_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    for family in FAMILIES:
        with np.load(parent / f"{family}_features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            if str(data["checkpoint_sha256"]) != plan["checkpoints"][family]:
                raise ValueError("Checkpoint provenance differs")
            vectors[family] = data["post_mean"].copy()
    summaries = {}
    for spec in PROFILES:
        name, cap = spec["name"], spec["cap"]
        selected_classes = spec["classes"] or classes
        indices = select_indices(rows, selected_classes, cap, plan["selection_seed"], balanced=spec["balanced"])
        selected_rows = [rows[i] for i in indices]
        selected_labels, selected_ids = labels[indices], ids[indices]
        path = out / name
        path.mkdir(exist_ok=True)
        write_rows(path / "samples.csv", selected_rows)
        sources = np.array([audit["source_path_by_sample_id"][ident] for ident in selected_ids])
        if len(set(sources)) != len(indices):
            raise ValueError("Repeated original recordings")
        print(f"Locked {name}: {len(indices)} real clips, {dict(Counter(selected_labels))}", flush=True)
        evaluated, fits, fit_info = {}, {}, []
        for family in FAMILIES:
            features = vectors[family][indices]
            metric = score(features, selected_labels, sources)
            evaluated[family] = {**prior["selected"][family], "metrics": metric}
            np.savez_compressed(path / f"{family}_features.npz", post_mean=features,
                                sample_ids=selected_ids, labels=selected_labels,
                                checkpoint_sha256=plan["checkpoints"][family])
            fits[family] = {}
            for perplexity, seed in FIT_SETTINGS:
                filename = path / f"{family}_cosine_p{perplexity}_s{seed}.npz"
                if args.resume and filename.exists():
                    with np.load(filename, allow_pickle=False) as data:
                        np.testing.assert_array_equal(data["sample_ids"], selected_ids)
                        if str(data["features_sha256"]) != digest(path / f"{family}_features.npz"):
                            raise ValueError("Stale coordinate cache")
                        xy, kl, trust = data["coordinates"].copy(), float(data["kl_divergence"]), float(data["trustworthiness"])
                else:
                    xy, kl, trust = fit_cosine(features, perplexity, seed)
                    np.savez_compressed(filename, coordinates=xy, labels=selected_labels, sample_ids=selected_ids,
                                        kl_divergence=kl, trustworthiness=trust,
                                        features_sha256=digest(path / f"{family}_features.npz"))
                fits[family][(perplexity, seed)] = xy
                fit_info.append({"family": family, "perplexity": perplexity, "seed": seed,
                                 "kl_divergence": kl, "trustworthiness_k10": trust})
                save_json(path / "fits.json", fit_info)
                print(f"{name}/{family}: P={perplexity}, seed={seed}, KL={kl:.4f}, trust={trust:.3f}", flush=True)
        report = {"samples": len(indices), "classes": len(selected_classes), "class_counts": dict(Counter(selected_labels)),
                  "split": "train+val", "split_counts": dict(Counter(audit["split_by_sample_id"][ident] for ident in selected_ids)),
                  "selected": evaluated, "labels_reviewed": False, "independent_test": False,
                  "post_hoc_focus": bool(spec["classes"]), "selection_rationale": spec["rationale"],
                  "sample_selection_uses_model_features": False, "original_recordings": len(set(sources)),
                  "display": {"scope_label": spec["scope_label"], "data_label": "HumanML3D train + validation",
                              "marker_size": 31 if not spec["balanced"] else 44,
                              "embedding_label": "Cosine t-SNE on original 512-D features | P = 30, seed = 42",
                              "note": "Exploratory action subset selected after earlier diagnostics; caption labels await review."
                                      if spec["classes"] else "Same samples on both sides; one clip per original recording; caption labels await review."}}
        coordinates = {family: fits[family][(30, 42)] for family in FAMILIES}
        for annotated in (False, True):
            plotted = render(path, selected_classes, selected_rows, report, coordinates, annotated=annotated)
            if plotted != {family: len(indices) for family in FAMILIES}:
                raise AssertionError("Missing plotted points")
        sensitivity(path, selected_classes, selected_labels, fits)
        save_json(path / "report.json", report)
        summaries[name] = report
    action_differences(prior, classes, out)
    save_json(out / "report.json", {"profiles": summaries, "full_cohort_reference": str(parent / "report.json"),
              "new_training": False, "new_test_evaluation": False, "teacher_features_used": False,
              "unchanged_source_sha256": inputs})
    if inputs != {path: digest(path) for path in inputs}:
        raise AssertionError("An original artifact changed")
    lines = ["# 适中点数与重点动作展示", "", "固定现有两组权重与量化后时间均值特征；仅改变展示样本构成及降维距离。未启动训练，未加入新的测试运动。", "",
             "本轮设计发生在先前诊断之后，属于探索性展示；重点类别的选择参考了先前收益，不能作为独立检验或所有动作均优越的证据。", "",
             "|视图|点数|动作数|普通Silhouette|TMR Silhouette|普通3-NN|TMR 3-NN|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, report in summaries.items():
        a, b = (report["selected"][family]["metrics"] for family in FAMILIES)
        lines.append(f"|{name}|{report['samples']}|{report['classes']}|{a['macro_silhouette']:.4f}|{b['macro_silhouette']:.4f}|{a['macro_neighbor_purity']:.2%}|{b['macro_neighbor_purity']:.2%}|")
    lines += ["", "overview_490保留全部10类，每类最多60段；较少类别保留全部。focused_200为跑、跳、坐、踢、投掷各40段的重点动作示例，动作是在此前诊断后确定的。",
              "样本按seed20261029及样本ID的SHA256排序，选样函数不接触模型特征、分数或二维坐标。每个原始录制至多一个片段；两侧完全相同。",
              "", "降维改为在原始512维上使用余弦距离，依据是既有定量诊断也使用余弦距离。此前主图使用PCA50再欧氏t-SNE；因此两版图不能只凭外观归因于模型变化。",
              "主图仍固定perplexity30、seed42、PCA初始化、2000次迭代。两侧均报告P30/50×seed0/42的完整敏感性结果，没有按图形效果挑主图种子。",
              "", "action_differences展示原2050段参考集的全部10类指标变化，同时保留变差的动作；其统计样本与子集图不同。",
              "PNG600dpi；PDF/SVG为矢量；preview为轻量预览。所有数值指标来自原始512维特征，不从二维图计算。标签仍待核验，数据含训练集。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(out / "verification.json", {"status": "complete", "new_training": False,
        "profiles": {name: {"samples": r["samples"], "original_recordings": r["original_recordings"]} for name, r in summaries.items()},
        "code_sha256": {name: digest(PROJECT / name) for name in ("optimize_display.py", "render_polished.py")},
        "output_sha256": {str(path.relative_to(out)): digest(path) for path in out.rglob("*") if path.suffix in (".png", ".pdf", ".svg", ".npz")}})
    print(json.dumps({"output": str(out), "samples": {name: r["samples"] for name, r in summaries.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
