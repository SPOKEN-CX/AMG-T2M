#!/usr/bin/env python3
"""Restyle the latest scientific comparison using the user's two-panel reference."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import FancyBboxPatch

from common import PROJECT, digest, read_rows, save_json
from render_polished import FAMILIES, PALETTE


INK = "#26313D"
MUTED = "#6B7682"
ACCENT = "#147C85"


def metric_label(fig, center, metric):
    parts = [
        ("Silhouette", 8.6, INK, False),
        (f"{metric['macro_silhouette']:.4f}", 9.3, ACCENT, True),
        (" | ", 8.0, "#9AA3AD", False),
        ("3-NN purity", 8.6, INK, False),
        (f"{metric['macro_neighbor_purity']:.2%}", 9.3, ACCENT, True),
    ]
    children = [TextArea(text, textprops={"fontfamily": "STIXGeneral", "fontsize": size,
                    "color": color, "weight": "bold" if bold else "normal"})
                for text, size, color, bold in parts]
    packed = HPacker(children=children, align="center", pad=0, sep=4)
    artist = AnnotationBbox(packed, (center, .102), xycoords=fig.transFigure,
                           box_alignment=(.5, .5), frameon=False, pad=0)
    fig.add_artist(artist)
    return artist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT / "artifacts/all_fid_audit_700_v1/legacy_lowest_record_vs_netlast_700")
    parser.add_argument("--reference", type=Path, default=PROJECT / "artifacts/comparison_1200_v1/comparison_polished_annotated.pdf")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--right-title", help="Override the displayed right-panel title only")
    parser.add_argument("--hide-counts", action="store_true", help="Show action names without the sample-count column")
    parser.add_argument("--point-size", type=float, default=6.4, help="Scatter marker area in points squared")
    parser.add_argument("--compact-panels", action="store_true", help="Use taller panels and tighter bounds")
    args = parser.parse_args()
    if not 0 < args.point_size <= 40:
        raise ValueError("Point size must be greater than zero and at most 40")
    study, reference = args.study.resolve(), args.reference.resolve()
    out = args.output.resolve() if args.output else study / "polished_clean_v1"
    if out.exists():
        raise FileExistsError("Preserve prior figures; use a new output directory")
    report = json.loads((study / "report.json").read_text())
    design = report["journal_design"]
    rows = read_rows(study / "samples.csv")
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    classes = list(report["class_counts"])
    counts = Counter(labels)
    assert len(rows) == len(set(ids)) == report["samples"]
    assert dict(counts) == report["class_counts"]
    assert set(classes) <= set(PALETTE)
    inputs = {str(path): digest(path) for path in (study / "report.json", study / "samples.csv", reference)}
    coordinates = {}
    for family in FAMILIES:
        path = study / design["coordinate_files"][family]
        inputs[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            xy = data["coordinates"].copy()
            assert xy.shape == (len(rows), 2) and np.isfinite(xy).all()
        coordinates[family] = xy

    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 9,
        "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": False, "savefig.facecolor": "white"})
    # Preserve the reference's wide composition at half its physical dimensions.
    physical_size = (8.4, 4.0) if args.compact_panels else (8.4, 3.05)
    fig = plt.figure(figsize=physical_size, facecolor="white")
    order = np.random.default_rng(42).permutation(len(rows))
    colors = np.array([PALETTE[label] for label in labels])
    axes, metric_boxes, titles = {}, [], []
    title_strings = list(design.get("titles", ["(a) Baseline VQ-VAE", "(b) TMR-guided VQ-VAE"]))
    if args.right_title:
        title_strings[1] = args.right_title
    for family, left, title in zip(FAMILIES, (.022, .440), title_strings):
        ax = fig.add_axes((left, .205, .400, .670))
        xy = coordinates[family]
        center = (xy.min(0)+xy.max(0))/2
        span = np.ptp(xy, axis=0)
        pad = .530 if args.compact_panels else .555
        ax.set_xlim(center[0]-pad*span[0], center[0]+pad*span[0])
        ax.set_ylim(center[1]-pad*span[1], center[1]+pad*span[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
            boxstyle="round,pad=0,rounding_size=0.007", facecolor="none",
            edgecolor="#C4CCD4", linewidth=.55, linestyle=(0, (3.4, 2.8)),
            clip_on=False, zorder=0))
        points = ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=args.point_size,
            alpha=.93, edgecolors="white", linewidths=.15, rasterized=False, zorder=3)
        np.testing.assert_array_equal(np.asarray(points.get_offsets()), xy[order])
        axes[family] = ax
        titles.append(fig.text(left+.200, .951, title, ha="center", va="center",
                               fontsize=11, fontweight="bold"))
        metric_boxes.append(metric_label(fig, left+.200, report["selected"][family]["metrics"]))

    legend = fig.add_axes((.858, .205, .122, .670))
    legend.set_axis_off()
    legend.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=legend.transAxes,
        boxstyle="round,pad=0,rounding_size=0.024", edgecolor="#D2D8DE",
        facecolor="#FCFDFE", linewidth=.6, clip_on=False))
    legend_text = [legend.text(.50 if args.hide_counts else .12, .937, "Action", fontsize=9.4,
                              fontweight="bold", va="center", ha="center" if args.hide_counts else "left")]
    if not args.hide_counts:
        legend_text.append(legend.text(.88, .937, "n", fontsize=8, color=MUTED, va="center", ha="right"))
    legend.plot([.11, .89], [.885, .885], color="#E0E5E9", lw=.55)
    row_text = []
    for label, y in zip(classes, np.linspace(.808, .103, len(classes))):
        legend.scatter([.20 if args.hide_counts else .15], [y], s=24, c=PALETTE[label], edgecolors="white", linewidths=.30)
        name = legend.text(.37 if args.hide_counts else .28, y, label, fontsize=9.1, va="center")
        legend_text.append(name)
        if not args.hide_counts:
            number = legend.text(.88, y, str(counts[label]), fontsize=8, color=MUTED,
                                 va="center", ha="right")
            legend_text.append(number)
            row_text.append((name, number))
    legend.set(xlim=(0, 1), ylim=(0, 1))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for family, ax in axes.items():
        origin, ex, ey = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_allclose(np.linalg.norm(ex-origin), np.linalg.norm(ey-origin), rtol=1e-6)
        bounds = ax.get_window_extent(renderer)
        pixels = ax.transData.transform(coordinates[family])
        assert (pixels[:, 0] > bounds.x0+1).all() and (pixels[:, 0] < bounds.x1-1).all()
        assert (pixels[:, 1] > bounds.y0+1).all() and (pixels[:, 1] < bounds.y1-1).all()
    for name, number in row_text:
        assert name.get_window_extent(renderer).x1 < number.get_window_extent(renderer).x0
    for title, metric, ax in zip(titles, metric_boxes, axes.values()):
        panel = ax.get_window_extent(renderer)
        for artist in (title, metric):
            box = artist.get_window_extent(renderer)
            assert box.x0 >= panel.x0 and box.x1 <= panel.x1, "Title or metric extends beyond its panel"
    for artist in [*titles, *metric_boxes, *legend_text]:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1

    out.mkdir(parents=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(out / f"comparison_clean.{suffix}")
    fig.savefig(out / "comparison_clean.png", dpi=900)
    fig.savefig(out / "comparison_clean_preview.png", dpi=300)
    plt.close(fig)
    with (out / "plotted_coordinates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "label", "baseline_x", "baseline_y", "tmr_x", "tmr_y"])
        for index, row in enumerate(rows):
            writer.writerow([row["sample_id"], row["label"],
                             *[float(value) for value in coordinates["baseline"][index]],
                             *[float(value) for value in coordinates["tmr_b512"][index]]])

    # Experimental provenance follows the source weights, independently of display names.
    legacy = "legacy" in report["selected"]["tmr_b512"]["name"].lower()
    baseline_name = report["selected"]["baseline"]["name"]
    projection_caption = design.get("projection_caption",
        "PCA to 50 dimensions and Euclidean t-SNE are fitted independently using "
        "the same settings (perplexity 30, early exaggeration 12, seed 42, 3000 iterations). ")
    caption = (
        f"Latent-space comparison on the same {len(rows):,} HumanML3D train and validation clips "
        f"across {len(classes)} caption-derived action categories. One 64-frame crop represents each "
        f"original recording. The left panel uses the ordinary VQ-VAE {baseline_name} checkpoint; "
        + ("the right panel uses the historical contrastive VQ-VAE net_best_fid checkpoint. " if legacy
           else "the right panel uses the TMR-guided VQ-VAE checkpoint specified in the source report. ")
        + "For each model, 512-dimensional features are formed by averaging codebook embeddings over time "
        "and L2 normalization. "
        + projection_caption
        + "Silhouette and 3-NN purity are calculated in the original 512-dimensional space and macro-averaged "
        "across actions. Checkpoints and action categories were selected exploratorily; caption-derived "
        "labels await human review. This comparison does not establish a controlled TMR ablation. "
        + ("The legacy directory reports a historical minimum FID of 0.0301; available old source points "
           "to test-based evaluation, but the original command and source snapshot are missing. The "
           "weight file has no iteration metadata, and exact identity with the logged 184000-step state "
           "is unverified. Its historical FID must not be compared directly with the current AMG validation FID."
           if legacy else "")
    )
    # Keep experimental qualifications in the figure caption, not as small chart text.
    (out / "FIGURE_CAPTION.txt").write_text(caption+"\n", encoding="utf-8")
    (out / "FIGURE_NOTES.zh-CN.md").write_text(
        "# 精简排版版\n\n"
        "参考用户指定comparison_polished_annotated.pdf，改为左右双面板、右侧竖排共享图例。"
        + ("图内保留模型标题、动作名称、轮廓系数和3-NN纯度，按用户要求隐藏样本数量及n列。" if args.hide_counts
           else "图内保留模型标题、动作与数量、轮廓系数和3-NN纯度。")
        + "删除总标题、重复checkpoint/FID/步数、投影参数与多行底部小字。必要实验信息移到FIGURE_CAPTION.txt和本说明。"
        + ("右侧显示名称按用户要求修改为"+title_strings[1]+"；旧版权重的身份与来源说明保留在图注中。\n\n" if args.right_title
           else "旧版模型的Legacy标识保留。\n\n")
        + "两图点大小、透明度、描边和绘制顺序完全一致，原始二维坐标、样本、类别、权重和指标逐项保留。"
        f"每侧{len(rows)}个矢量散点；PNG为900dpi（{round(physical_size[0]*900)}×{round(physical_size[1]*900)}），PDF/SVG可缩放。预览PNG为300dpi。\n\n"
        + "源数据目录："+str(study)+"\n\n"
        + "参考版式文件："+str(reference)+"\n\n"
        + "实验范围："+design.get("scope_note", "见源report.json")+"\n\n"
        + f"{len(rows)}段来自train+val且字幕标签待核验，动作和权重经过开发筛选，属于探索性对比。"
        "图中没有通过移动、删减、复制或合成点强化分离。\n\n"
        + "英文图注：\n\n"+caption+"\n", encoding="utf-8")
    assert inputs == {path: digest(path) for path in inputs}
    save_json(out / "render_manifest.json", {
        "operation": "Style-only rendering of saved coordinates; no feature extraction or t-SNE refit",
        "reference_pdf": str(reference), "source_study": str(study), "input_sha256": inputs,
        "renderer_sha256": digest(Path(__file__)), "samples_per_panel": len(rows),
        "class_counts": dict(counts), "source_selected_checkpoints": report["selected"],
        "source_scientific_scope": design.get("scope"), "legacy_identity_retained_in_caption": legacy,
        "displayed_titles": title_strings, "legend_sample_counts_displayed": not args.hide_counts,
        "coordinate_changes": False, "samples_cropped": False, "isotropic_geometry_verified": True,
        "marker_size": args.point_size, "marker_alpha": .93, "marker_white_edge_width": .15,
        "both_panels_same_marker_style_and_draw_order": True, "labels_and_metrics_nonoverlapping": True,
        "physical_size_inches": list(physical_size), "compact_panels": args.compact_panels, "png_dpi": 900, "vector_pdf_and_svg": True,
        "source_explanations_moved_to": "FIGURE_CAPTION.txt",
        "output_sha256": {path.name: digest(path) for path in sorted(out.iterdir())}})
    print(json.dumps({"output": str(out), "samples_per_panel": len(rows),
                      "coordinates_unchanged": True, "png_dpi": 900}, ensure_ascii=False))


if __name__ == "__main__":
    main()
