#!/usr/bin/env python3
"""Restyle the completed validation comparison using its unchanged t-SNE coordinates."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import PROJECT, digest, read_rows, save_json


PALETTE = {
    "walk": "#E34C4C", "run": "#2478CA", "jump": "#26A450",
    "sit": "#8A43B9", "kick": "#F08A28", "wave": "#E76C9E",
    "clap": "#D7AD26", "throw": "#9A7059", "crawl": "#08A7AA",
    "squat": "#92B83B",
    "dance": "#00A0AC", "arm_raise": "#5968A8", "bend": "#A68A64",
    "arm_circle": "#B64F79", "stand_up": "#6B8C39", "stairs": "#695C4E",
    "punch": "#6B4C9A", "push": "#A34A3A", "stretch": "#B88722",
}
FAMILIES = ("baseline", "tmr_b512")


def load_saved(study):
    probe = study / "checkpoint_probe"
    paths = [study / "samples.csv", study / "config_snapshot.json", probe / "report.json"]
    rows = read_rows(paths[0])
    cfg, report = (json.loads(path.read_text()) for path in paths[1:])
    if report["cohort_sha256"] != digest(paths[0]):
        raise ValueError("The report and sample manifest differ")
    if report["split"] != "val" or report["feature"] != "post_mean":
        raise ValueError("Expected the completed post_mean validation comparison")
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    if len(set(ids)) != len(ids) or len(rows) != report["samples"]:
        raise ValueError("Duplicate or missing clips")
    coordinates = {}
    for family in FAMILIES:
        path = probe / f"post_mean_{family}_coordinates.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            xy = data["coordinates"].copy()
            if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
                raise ValueError("Invalid coordinates")
            if not np.isfinite(data["kl_divergence"]).all():
                raise ValueError("Invalid saved t-SNE fit")
        path = probe / f"{family}_selected_features.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            if str(data["checkpoint_sha256"]) != report["selected"][family]["sha256"]:
                raise ValueError("The selected features belong to another checkpoint")
            features = data["post_mean"]
            if features.shape != (len(rows), 512) or not np.isfinite(features).all():
                raise ValueError("Invalid saved features")
        coordinates[family] = xy
    classes = [item["name"] for item in cfg["classes"]]
    if set(classes) != set(labels) or not set(classes) <= set(PALETTE):
        raise ValueError("The plotting palette does not cover the cohort")
    return rows, classes, report, coordinates, {str(path): digest(path) for path in paths}


def style_panel(ax, xy, labels, classes, order, marker_size=72):
    # Only the view limits change. Equal aspect preserves every point's geometry.
    center = (xy.max(0) + xy.min(0)) / 2
    span = np.ptp(xy, axis=0)
    ax.set_xlim(center[0] - span[0] * .575, center[0] + span[0] * .575)
    ax.set_ylim(center[1] - span[1] * .575, center[1] + span[1] * .575)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.008",
        transform=ax.transAxes, fc="none", ec="#C5CBD3", lw=.85,
        linestyle=(0, (3.5, 3)), clip_on=False, zorder=0,
    ))
    # A fixed shuffle avoids consistently hiding the same class behind another.
    colors = np.array([PALETTE[label] for label in labels])
    dots = ax.scatter(xy[order, 0], xy[order, 1], s=marker_size, c=colors[order],
                      alpha=.88, edgecolors="white", linewidths=.55 if marker_size > 35 else .18, zorder=3)
    np.testing.assert_array_equal(np.asarray(dots.get_offsets()), xy[order])
    return len(dots.get_offsets())


def render(output, classes, rows, report, coordinates, *, annotated=False, dpi=600):
    labels = np.array([row["label"] for row in rows])
    counts = Counter(labels)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 12,
        "text.color": "#212934", "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": False, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(16.8, 6.1), facecolor="white")
    display = report.get("display", {})
    marker_size = display.get("marker_size", 72)
    if display.get("scope_label"):
        fig.text(.5, .975, display["scope_label"], ha="center", va="center", fontsize=9.4,
                 color="#64717F", fontfamily="DejaVu Sans")
    bottom, height = (.22, .64) if annotated else (.145, .715)
    positions = ((.022, bottom, .400, height), (.440, bottom, .400, height))
    axes = [fig.add_axes(position) for position in positions]
    titles = ("(a) Baseline VQ-VAE", "(b) Ours (TMR-guided VQ-VAE)")
    order = np.random.default_rng(42).permutation(len(rows))
    plotted_counts = {}
    for ax, family, title in zip(axes, FAMILIES, titles):
        plotted_counts[family] = style_panel(ax, coordinates[family], labels, classes, order, marker_size)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=17)
        if annotated:
            metric = report["selected"][family]["metrics"]
            center = ax.get_position().x0 + ax.get_position().width / 2
            fig.text(center, .164, f"Silhouette  {metric['macro_silhouette']:.4f}"
                     f"     |     3-NN purity  {metric['macro_neighbor_purity']:.2%}",
                     ha="center", va="center", fontsize=11.2)
            checkpoint = report["selected"][family]["name"]
            fig.text(center, .119, f"Checkpoint: {checkpoint}", ha="center", va="center",
                     fontsize=9.4, color="#737D89", fontfamily="DejaVu Sans")

    legend = fig.add_axes((.858, bottom, .122, height))
    legend.set_axis_off()
    legend.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, transform=legend.transAxes,
        boxstyle="round,pad=0,rounding_size=0.025", ec="#D3D8DF", fc="#FCFCFD",
        linewidth=.8, clip_on=False,
    ))
    legend.text(.12, .947, "Action", fontsize=12.5, weight="bold", va="center")
    legend.text(.88, .947, "n", fontsize=10.5, color="#7B8590", va="center", ha="right")
    legend.plot([.11, .89], [.903, .903], color="#E0E4EA", lw=.7)
    for label, y in zip(classes, np.linspace(.847, .09, len(classes))):
        legend.scatter([.15], [y], s=91, c=PALETTE[label], edgecolor="white", linewidth=.6)
        legend.text(.28, y, label, fontsize=12.5, va="center")
        legend.text(.88, y, str(counts[label]), fontsize=10.2, color="#747E89",
                    ha="right", va="center", fontfamily="DejaVu Sans")
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)

    data_label = display.get("data_label", "HumanML3D validation")
    fig.text(.022, .064, f"{data_label}  |  {len(rows):,} clips  |  {len(classes)} actions"
             "  |  64 frames / clip", fontsize=9.1, color="#707A85", fontfamily="DejaVu Sans")
    embedding_label = display.get("embedding_label", "Post-quantization mean  |  t-SNE: P = 30, seed = 42")
    fig.text(.980, .064, embedding_label,
             ha="right", fontsize=9.1, color="#707A85", fontfamily="DejaVu Sans")
    note = display.get("note", "Exploratory validation comparison; caption labels await review.")
    fig.text(.022, .025, note,
             fontsize=8.4, color="#858D97", fontfamily="DejaVu Sans")
    if annotated:
        fig.text(.980, .025, "Metrics in the original feature space; macro-averaged across actions.",
                 ha="right", fontsize=8.4, color="#858D97", fontfamily="DejaVu Sans")
    stem = "comparison_polished_annotated" if annotated else "comparison_polished"
    # Also save a light preview for convenient inline display. The PDF/SVG are vector.
    fig.savefig(output / f"{stem}_preview.png", dpi=180)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=dpi)
    plt.close(fig)
    return plotted_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT / "artifacts/expanded_validation_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    study = args.study.resolve()
    output = args.output.resolve() if args.output else study / "checkpoint_probe/polished"
    if args.dpi < 150:
        raise ValueError("Use at least 150 dpi")
    rows, classes, report, coordinates, inputs = load_saved(study)
    output.mkdir(parents=True, exist_ok=True)
    count = render(output, classes, rows, report, coordinates, dpi=args.dpi)
    annotated_count = render(output, classes, rows, report, coordinates, annotated=True, dpi=args.dpi)
    if count != {family: len(rows) for family in FAMILIES} or annotated_count != count:
        raise AssertionError("Some samples were not rendered")
    with (output / "plotted_coordinates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["sample_id", "label", "baseline_x", "baseline_y", "tmr_b512_x", "tmr_b512_y"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"sample_id": row["sample_id"], "label": row["label"],
                             **{f"{family}_{axis}": float(coordinates[family][i, j])
                                for family in FAMILIES for j, axis in enumerate(("x", "y"))}})
    if inputs != {path: digest(path) for path in inputs}:
        raise AssertionError("A source artifact changed while rendering")
    manifest = {
        "operation": "Restyle saved coordinates; no feature extraction or t-SNE refit",
        "split": "val", "labels_reviewed": all(row["review_status"] == "accepted" for row in rows),
        "sample_count_per_panel": count, "class_counts": dict(Counter(row["label"] for row in rows)),
        "feature": "post_mean", "selected_checkpoints": report["selected"],
        "input_sha256": inputs, "renderer_sha256": digest(Path(__file__)),
        "palette": PALETTE, "dpi": args.dpi, "size_inches": [16.8, 6.1],
        "coordinate_change": "None: exact original coordinates; equal aspect; fixed shuffled draw order",
        "output_sha256": {path.name: digest(path) for path in sorted(output.iterdir())
                          if path.suffix in (".png", ".pdf", ".svg", ".csv")},
    }
    save_json(output / "render_manifest.json", manifest)
    (output / "FIGURE_NOTES.zh-CN.md").write_text(
        "# 美化版真实潜空间对比图\n\n"
        "主图为 comparison_polished；annotated 版本增加指标和权重名称。PNG 为 600 dpi（默认），"
        "PDF/SVG 为矢量文件，preview 为便于预览的低分辨率副本。\n\n"
        "使用已有10类108段验证运动，左侧普通VQ-VAE net_best_fid，右侧TMR引导VQ-VAE "
        "model_50000。两侧各108个真实片段，动作与样本ID完全一致。原始t-SNE坐标逐点保留，"
        "没有重新挑参数或样本。一个点是一段运动，未通过抖动、重复或合成点增加密度。\n\n"
        "美化包括动作配色、白色细描边与透明度、增大散点、衬线标题、等尺寸面板、细虚线边框、"
        "共享带数量图例、减少留白和说明文字。等比例坐标保留二维几何。\n\n"
        "原始512维特征的宏平均Silhouette为0.0463→0.0915，宏平均3-NN纯度59.70%→60.68%。"
        "这些值不由二维图计算。分离效果仍由实际特征决定，美化不会把交叠动作变成分开的簇。\n\n"
        "这是验证集探索图，标签待人工核验；现有两组训练设置不完全一致，不能视为严格的TMR因果消融。"
        "教师特征没有作为我们的VQ-VAE特征展示。完整来源和输出哈希见render_manifest.json，"
        "所有点的数值见plotted_coordinates.csv。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "samples_per_panel": count,
                      "dpi": args.dpi, "coordinates_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
