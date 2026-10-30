#!/usr/bin/env python3
"""Show real cosine similarities directly, without a nonlinear 2D projection."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import PROJECT, digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROJECT / "artifacts/moderate_comparison_v1/focused_200")
    args = parser.parse_args()
    out = args.profile.resolve()
    report = json.loads((out / "report.json").read_text())
    result, matrices = {}, {}
    expected_ids, expected_labels = None, None
    for family in ("baseline", "tmr_b512"):
        path = out / f"{family}_features.npz"
        with np.load(path, allow_pickle=False) as data:
            x, ids, labels = data["post_mean"], data["sample_ids"], data["labels"]
            if expected_ids is None:
                expected_ids, expected_labels = ids.copy(), labels.copy()
            else:
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_array_equal(labels, expected_labels)
            if str(data["checkpoint_sha256"]) != report["selected"][family]["sha256"]:
                raise ValueError("Unexpected checkpoint")
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
            similarity = np.clip(x @ x.T, -1, 1)
        same = labels[:, None] == labels[None, :]
        off = ~np.eye(len(labels), dtype=bool)
        within = float(similarity[same & off].mean())
        between = float(similarity[~same].mean())
        result[family] = {"within_class_mean_cosine": within, "between_class_mean_cosine": between,
                          "within_minus_between": within - between,
                          "features_sha256": digest(path), "same_class_off_diagonal_pairs": int((same & off).sum()),
                          "different_class_pairs": int((~same).sum())}
        matrices[family] = similarity
    classes = list(report["class_counts"])
    order = np.concatenate([np.flatnonzero(expected_labels == name) for name in classes])
    counts = [report["class_counts"][name] for name in classes]
    boundaries = np.cumsum([0] + counts)
    centers = (boundaries[:-1] + boundaries[1:] - 1)/2
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14.4, 7.8), facecolor="white")
    titles = ("(a) Baseline VQ-VAE", "(b) Ours (TMR-guided VQ-VAE)")
    bottom, height = (.265, .62) if len(classes) > 5 else (.225, .665)
    for family, x0, title in zip(("baseline", "tmr_b512"), (.07, .51), titles):
        ax = fig.add_axes((x0, bottom, .36, height))
        matrix = matrices[family][np.ix_(order, order)]
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest", aspect="equal")
        ax.set_xticks(centers, classes, rotation=0 if len(classes) <= 5 else 45,
                      fontsize=10.8 if len(classes) <= 5 else 8.4)
        ax.set_yticks(centers, classes, fontsize=10.8 if len(classes) <= 5 else 9.2)
        ax.tick_params(length=0, pad=8)
        for boundary in boundaries[1:-1] - .5:
            ax.axvline(boundary, color="white", lw=.65, alpha=.8)
            ax.axhline(boundary, color="white", lw=.65, alpha=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=14.3, weight="bold", pad=17)
        values = result[family]
        fig.text(x0+.18, .16, f"Within action: {values['within_class_mean_cosine']:.3f}"
                 f"   |   Between actions: {values['between_class_mean_cosine']:.3f}",
                 ha="center", fontsize=10.5, color="#546271")
        fig.text(x0+.18, .105, f"Similarity margin  {values['within_minus_between']:.3f}",
                 ha="center", fontsize=15.8, color="#167E75" if family == "tmr_b512" else "#536373", weight="bold")
    cbar = fig.colorbar(im, cax=fig.add_axes((.905, .285, .016, .55)), ticks=[-1, -.5, 0, .5, 1])
    cbar.set_label("Cosine similarity", labelpad=10, fontsize=11)
    cbar.outline.set_visible(False)
    scope = "Focused action subset; exploratory" if report["post_hoc_focus"] else "10-action overview; descriptive"
    fig.text(.5, .975, f"ORIGINAL 512-D FEATURE SIMILARITY | {report['samples']} clips | {scope}",
             ha="center", va="center", fontsize=9.3, color="#677584", fontfamily="DejaVu Sans")
    fig.text(.07, .045, "Same samples and color scale on both sides; diagonal self-similarities excluded from means.",
             fontsize=8.8, color="#718090", fontfamily="DejaVu Sans")
    fig.text(.07, .017, "HumanML3D train + validation | caption labels pending review | unchanged VQ-VAE features",
             fontsize=8.4, color="#7D8997", fontfamily="DejaVu Sans")
    fig.savefig(out / "similarity_structure_preview.png", dpi=180)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"similarity_structure.{suffix}", dpi=600)
    plt.close(fig)
    save_json(out / "similarity_structure_report.json", {
        "samples": report["samples"], "post_hoc_focus": report["post_hoc_focus"], "models": result,
        "similarity_margin_definition": "Mean off-diagonal same-action cosine minus mean different-action cosine, all sample pairs",
        "shared_color_limits": [-1, 1], "self_similarity_excluded": True,
        "original_features_unchanged": True, "renderer_sha256": digest(Path(__file__)),
        "output_sha256": {p.name: digest(p) for p in out.glob("similarity_structure.*")}})
    print(json.dumps({"profile": str(out), "models": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
