#!/usr/bin/env python3
"""Faithful vector realization of the GPT image model's journal-style design."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle

from common import PROJECT, digest, read_rows, save_json
from render_polished import FAMILIES, PALETTE


INK = "#23313F"
MUTED = "#6A7784"
TEAL = "#087782"


def text_area(text, size=8, *, color=INK, bold=False):
    return TextArea(text, textprops={"fontfamily": "STIXGeneral", "fontsize": size,
                    "color": color, "weight": "bold" if bold else "normal"})


def packed_label(fig, xy, children, sep=4):
    pack = HPacker(children=children, align="center", pad=0, sep=sep)
    box = AnnotationBbox(pack, xy, xycoords=fig.transFigure, box_alignment=(.5, .5),
                         frameon=False, pad=0)
    fig.add_artist(box)
    return box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT / "artifacts/comparison_1200_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    out = args.output.resolve() if args.output else study / "journal_style_v1"
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((study / "report.json").read_text())
    design = report.get("journal_design", {})
    rows = read_rows(study / "samples.csv")
    ids = np.array([r["sample_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    classes = list(report["class_counts"])
    if len(rows) != report["samples"] or len(set(ids)) != len(rows):
        raise ValueError("Unexpected sample count or duplicate ID")
    coordinates, hashes = {}, {str(study / "report.json"): digest(study / "report.json"),
                               str(study / "samples.csv"): digest(study / "samples.csv")}
    for family in FAMILIES:
        filename = design.get("coordinate_files", {}).get(family, f"{family}_pca_euclidean_p30_s42.npz")
        path = study / filename
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            xy = data["coordinates"].copy()
            if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
                raise ValueError("Invalid source coordinates")
        coordinates[family] = xy
        hashes[str(path)] = digest(path)
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 8,
                         "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "axes.unicode_minus": False})
    # A double-column-sized layout, with font sizes defined in physical points.
    fig = plt.figure(figsize=(7.2, 3.35), facecolor="white")
    context = design.get("context", f"HumanML3D  |  {len(rows):,} paired motion clips  |  {len(classes)} actions")
    fig.text(.5, .975, context,
             ha="center", va="center", fontsize=7.3, color=MUTED)
    order = np.random.default_rng(42).permutation(len(rows))
    colors = np.array([PALETTE[label] for label in labels])
    axes, point_artists, metric_boxes = {}, {}, []
    titles = design.get("titles", ("(a)  Baseline VQ-VAE", "(b)  TMR-guided VQ-VAE"))
    subtitles = design.get("subtitles", {})
    for family, x0, title in zip(FAMILIES, (.018, .518), titles):
        ax = fig.add_axes((x0, .326, .464, .522 if subtitles else .542))
        xy = coordinates[family]
        center = (xy.min(0)+xy.max(0))/2
        span = np.ptp(xy, axis=0)
        ax.set_xlim(center[0]-.575*span[0], center[0]+.575*span[0])
        ax.set_ylim(center[1]-.575*span[1], center[1]+.575*span[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_color("#B7C0C8")
            spine.set_linewidth(.38)
        # The two panels share all marker styling; only class color differs.
        dots = ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=design.get("marker_size", 3.3),
                          alpha=.90, edgecolors="white", linewidths=.07, rasterized=False)
        np.testing.assert_array_equal(np.asarray(dots.get_offsets()), xy[order])
        axes[family], point_artists[family] = ax, dots
        fig.text(x0+.232, .920, title, ha="center", va="center", fontsize=10.5, weight="bold")
        if family in subtitles:
            fig.text(x0+.232, .877, subtitles[family], ha="center", va="center", fontsize=6.2, color=MUTED)
        metric = report["selected"][family]["metrics"]
        metric_boxes.append(packed_label(fig, (x0+.232, .267), [
            text_area("Silhouette", 8.1), text_area(f"{metric['macro_silhouette']:.4f}", 9.0, color=TEAL, bold=True),
            text_area("|", 8.2, color=MUTED), text_area("3-NN purity", 8.1),
            text_area(f"{metric['macro_neighbor_purity']:.2%}", 9.0, color=TEAL, bold=True)], sep=3.5))
    legend_boxes = []
    legend_columns = int(design.get("legend_columns", 3 if len(classes) == 6 else 5))
    for i, label in enumerate(classes):
        row_size = min(legend_columns, len(classes)-(i//legend_columns)*legend_columns)
        x0 = .5 + ((i % legend_columns)-(row_size-1)/2)/legend_columns
        y0 = .185 - (i // legend_columns)*.059
        marker = DrawingArea(6, 6, 0, 0)
        marker.add_artist(Circle((3, 3), 1.9, fc=PALETTE[label], ec="none"))
        legend_boxes.append(packed_label(fig, (x0, y0), [marker, text_area(label, 7.7),
                     text_area(f"({report['class_counts'][label]})", 7.0, color=MUTED)], sep=3))
    fig.add_artist(Line2D([.018, .982], [.086, .086], transform=fig.transFigure,
                          color="#B7C0C8", lw=.38))
    fig.text(.018, .051, design.get("footer_left", "Train + validation; caption labels pending review."),
             va="center", fontsize=6.2, color=MUTED)
    fig.text(.982, .051, design.get("footer_right", "PCA50 + t-SNE; perplexity 30, seed 42."),
             ha="right", va="center", fontsize=6.2, color=MUTED)
    fig.text(.5, .020, design.get("footer_bottom", "Metrics measured in the original 512-D latent space; macro-averaged across actions."),
             ha="center", va="center", fontsize=6.2, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for family, ax in axes.items():
        # Verify isotropic drawing and that the new frame does not crop samples.
        origin, ex, ey = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_allclose(np.linalg.norm(ex-origin), np.linalg.norm(ey-origin), rtol=1e-6)
        pixels = ax.transData.transform(coordinates[family])
        bounds = ax.get_window_extent(renderer)
        if not ((pixels[:, 0] >= bounds.x0).all() and (pixels[:, 0] <= bounds.x1).all()
                and (pixels[:, 1] >= bounds.y0).all() and (pixels[:, 1] <= bounds.y1).all()):
            raise AssertionError("A sample is cropped by the new layout")
    legend_rows = [legend_boxes[i:i+legend_columns] for i in range(0, len(legend_boxes), legend_columns)]
    for boxes in [*legend_rows, metric_boxes]:
        boxes = sorted([box.get_window_extent(renderer) for box in boxes], key=lambda b: b.x0)
        for left, right in zip(boxes, boxes[1:]):
            if left.x1 >= right.x0:
                raise AssertionError("Legend or metric labels overlap")
    fig.savefig(out / "journal_comparison_preview.png", dpi=300)
    fig.savefig(out / "journal_comparison.png", dpi=900)
    for suffix in ("pdf", "svg"):
        fig.savefig(out / f"journal_comparison.{suffix}")
    plt.close(fig)
    with (out / "plotted_coordinates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "baseline_x", "baseline_y", "tmr_x", "tmr_y"])
        for i, row in enumerate(rows):
            writer.writerow([row["sample_id"], row["label"], *[float(v) for v in coordinates["baseline"][i]],
                             *[float(v) for v in coordinates["tmr_b512"][i]]])
    if hashes != {path: digest(path) for path in hashes}:
        raise AssertionError("A source artifact changed")
    native = out / "gpt_design.png"
    save_json(out / "design_provenance.json", {
        "design_method": "Built-in GPT image design, followed by coordinate-preserving scientific rendering"
                         if native.exists() else "Layout inherited from the prior GPT-designed figure; original-coordinate rendering",
        "scientific_final": "journal_comparison; re-rendered from the unchanged original t-SNE coordinates",
        "gpt_design_is_layout_reference": True, "gpt_design_sha256": digest(native) if native.exists() else None,
        "prompt_file": "imagegen_prompt.txt" if (out / "imagegen_prompt.txt").exists() else None, "input_sha256": hashes,
        "scientific_scope": design.get("scope", "Fixed-checkpoint descriptive comparison"),
        "renderer_sha256": digest(Path(__file__)), "samples_per_panel": len(rows),
        "exact_original_coordinates": True, "both_panels_same_marker_style": True,
        "marker_size": design.get("marker_size", 3.3), "legend_columns": legend_columns,
        "equal_aspect_verified": True, "all_samples_within_frames": True,
        "legend_and_metric_labels_nonoverlapping": True, "class_palette": PALETTE,
        "physical_size_inches": [7.2, 3.35], "png_dpi": 900, "font": "STIXGeneral",
        "pdf_and_svg_keep_vector_points": True, "new_training": False,
        "output_sha256": {p.name: digest(p) for p in out.iterdir() if p.suffix in (".png", ".pdf", ".svg", ".csv", ".txt")}})
    intro = (f"已使用内置GPT画图模型编辑{len(rows)}点主图，设计稿为gpt_design.png，完整提示词见imagegen_prompt.txt。"
             "生成模型输出仅作视觉设计参考，不依赖其逐点科学保真性。\n\n") if native.exists() else (
             "本图沿用此前GPT画图模型生成的期刊版式，从原始坐标绘制；此目录保留科学复绘图及来源记录。\n\n")
    (out / "DESIGN_NOTES.zh-CN.md").write_text(
        "# GPT视觉设计与原坐标复绘\n\n"
        + intro +
        f"正式图journal_comparison按该设计从原始坐标复绘，保留两侧各{len(rows)}个点、全部类别、原始指标、"
        "模型权重、投影参数与相同绘制顺序。没有重新计算特征或t-SNE。"
        "主要改进为底部双行共享图例、统一STIX衬线字体、等宽双面板、细实线边框、"
        "统一指标排版、移除内部checkpoint路径和右侧卡片、适合双栏宽度的物理字号。\n\n"
        "PNG为900dpi（6480×3015），PDF/SVG中的散点和文字均保留为矢量。"
        "逐点原坐标一致、等比例、完整可见和图例不重叠均已检查，详见design_provenance.json。\n\n"
        "数据仍为HumanML3D训练加验证的描述性展示，标签待人工核验。视觉美化没有改变实证结论。\n"
        + design.get("scope_note", ""),
        encoding="utf-8")
    print(json.dumps({"output": str(out), "samples_per_panel": len(rows),
                      "coordinates_unchanged": True, "png_dpi": 900, "vector_exports": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
