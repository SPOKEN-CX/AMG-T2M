#!/usr/bin/env python3
"""Plot the actual frozen VQ paired with the user-specified GPT evaluation."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from threadpoolctl import threadpool_limits

from analyze import latent_metrics, paired_bootstrap
from checkpoint_contrast import projection
from common import PROJECT, REPO, digest, read_rows, repo_path, save_json, write_rows
from features import FrozenEncoder, motion_batch, normalize
from optimize_display import fit_cosine, score, select_indices
from render_polished import PALETTE


CLASSES = ["run", "jump", "sit", "kick", "throw", "clap"]
FAMILIES = ("baseline", "tmr_b512")
SETTINGS = [(p, 12, s) for p in (20, 30, 50) for s in (0, 1, 42)]
PRIMARY = (30, 12, 42)
SEED = 20261029


def filename(family, setting):
    p, exaggeration, seed = setting
    return f"{family}_pca_p{p}_e{exaggeration}_s{seed}.npz"


def grid(out, labels, fits, settings, classes=None):
    classes = CLASSES if classes is None else classes
    fig, axes = plt.subplots(len(settings), 2, figsize=(8, 2.1*len(settings)), squeeze=False)
    for row, setting in enumerate(settings):
        for column, family in enumerate(FAMILIES):
            ax, xy = axes[row, column], fits[family][setting]
            for label in classes:
                points = xy[labels == label]
                ax.scatter(*points.T, s=8, color=PALETTE[label], alpha=.9, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xticks=[], yticks=[])
            ax.set_title(f"{'Baseline' if column == 0 else 'TMR, frozen VQ'} | P{setting[0]}, seed {setting[2]}", fontsize=9)
            for spine in ax.spines.values():
                spine.set_color("#B7C0C8")
    fig.suptitle(f"Same {len(labels)} clips and parameters | primary P30, seed42 | E12", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / "sensitivity.png", dpi=160)
    fig.savefig(out / "sensitivity.pdf")
    plt.close(fig)


def evidence(out, xs, labels, metrics, classes=None):
    classes = CLASSES if classes is None else classes
    class_count = len(classes)
    matrices, similarities = {}, {}
    for family, x in xs.items():
        gram = x @ x.T
        matrix = np.empty((class_count, class_count))
        for i, a in enumerate(classes):
            ia = np.flatnonzero(labels == a)
            for j, b in enumerate(classes):
                ib = np.flatnonzero(labels == b)
                block = gram[np.ix_(ia, ib)]
                matrix[i, j] = (block.sum()-np.trace(block))/(len(ia)*(len(ia)-1)) if i == j else block.mean()
        matrices[family] = matrix
        same = float(np.diag(matrix).mean())
        different = float(matrix[~np.eye(class_count, dtype=bool)].mean())
        similarities[family] = {"macro_same_class_cosine": same, "macro_different_class_cosine": different,
                                "gap": same-different, "matrix": matrix.tolist()}
    plt.rcParams.update({"font.family": "STIXGeneral", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.35), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    fig.subplots_adjust(left=.045, right=.96, bottom=.25, top=.79, wspace=.53)
    for ax, family, title in zip(axes[:2], FAMILIES, ("(a) Baseline similarities", "(b) TMR similarities")):
        im = ax.imshow(matrices[family], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(class_count), classes, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(class_count), classes, fontsize=7.5)
        ax.set_title(title, fontsize=10.5, weight="bold", pad=10)
        for i in range(class_count):
            for j in range(class_count):
                value = matrices[family][i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(value) > .6 else "#23313F")
        stats = similarities[family]
        bounds = ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2, .115,
                 f"Within {stats['macro_same_class_cosine']:.3f} | Between {stats['macro_different_class_cosine']:.3f}\nGap {stats['gap']:.3f}",
                 ha="center", va="center", fontsize=8)
    cax = fig.add_axes((.594, .30, .009, .40))
    fig.colorbar(im, cax=cax, ticks=[-1, 0, 1]).ax.tick_params(labelsize=7, length=2)
    delta = np.array([metrics["tmr_b512"]["per_class"][label]["silhouette"]-
                      metrics["baseline"]["per_class"][label]["silhouette"] for label in classes])
    ax = axes[2]
    ax.barh(np.arange(class_count), delta, height=.60, color=["#087782" if d >= 0 else "#B9624E" for d in delta])
    ax.axvline(0, lw=.7, color="#8794A0")
    ax.set_yticks(range(class_count), classes, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("(c) Silhouette change", fontsize=10.5, weight="bold", pad=10)
    ax.set_xlabel("TMR minus baseline", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    span = max(float(np.ptp(delta)), .03)
    ax.set_xlim(min(0, delta.min())-.28*span, max(0, delta.max())+.4*span)
    for i, value in enumerate(delta):
        ax.text(value+(.04*span if value >= 0 else -.04*span), i, f"{value:+.3f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.5, .94, f"Original 512-D evidence | {len(labels)} paired clips | {class_count} actions", ha="center", fontsize=13, weight="bold", color="#23313F")
    fig.text(.5, .035, "Cosine similarities use a shared [-1, 1] scale. Caption labels pending review; training-inclusive exploration.",
             ha="center", fontsize=7.5, color="#6A7784")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"original_space_evidence.{suffix}", dpi=400)
    plt.close(fig)
    save_json(out / "similarity_structure.json", similarities)


def cohort(out, rows, sources, vectors, states, settings, *, reference=False, classes=None):
    classes = CLASSES if classes is None else classes
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "samples.csv", rows)
    labels = np.array([r["label"] for r in rows])
    ids = np.array([r["sample_id"] for r in rows])
    assert len(set(sources)) == len(rows) == len(set(ids))
    metrics, fits, records, point_scores = {}, {}, [], {}
    for family in FAMILIES:
        x = vectors[family]
        metrics[family] = score(x, labels, sources)
        local = latent_metrics(x, labels, sources, 3)
        five = latent_metrics(x, labels, sources, 5)
        metrics[family]["macro_5nn_purity"] = float(np.mean([
            five["knn_purity_per_sample"][labels == label].mean() for label in classes]))
        point_scores[family] = local
        feature_path = out / f"{family}_features.npz"
        np.savez_compressed(feature_path, post_mean=x, sample_ids=ids, labels=labels,
                            checkpoint_sha256=states[family]["sha256"])
        pca = PCA(n_components=50, svd_solver="full")
        reduced = pca.fit_transform(x)
        np.savez_compressed(out / f"{family}_pca.npz", components=pca.components_, mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_)
        fits[family] = {}
        for setting in settings:
            xy, kl = projection(reduced, setting)
            fits[family][setting] = xy
            np.savez_compressed(out / filename(family, setting), coordinates=xy, sample_ids=ids, labels=labels,
                                kl_divergence=kl, features_sha256=digest(feature_path))
            records.append({"family": family, "setting": list(setting), "kl_divergence": kl,
                            "trustworthiness_original_cosine_k10": float(trustworthiness(x, xy, n_neighbors=10, metric="cosine")),
                            "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum())})
        print(f"{out.name}/{family}: silhouette={metrics[family]['macro_silhouette']:.4f}, 3-NN={metrics[family]['macro_neighbor_purity']:.2%}", flush=True)
    save_json(out / "fits.json", records)
    intervals = {key: paired_bootstrap(point_scores["baseline"][key], point_scores["tmr_b512"][key], labels, 2000, SEED)
                 for key in ("silhouette_per_sample", "knn_purity_per_sample")}
    intervals["aggregation"] = "Micro-average of fixed sample scores; equals the macro-average only in balanced cohorts"
    save_json(out / "conditional_sample_intervals.json", intervals)
    np.savez_compressed(out / "per_sample_metrics.npz", sample_ids=ids, labels=labels,
                        baseline_silhouette=point_scores["baseline"]["silhouette_per_sample"],
                        tmr_silhouette=point_scores["tmr_b512"]["silhouette_per_sample"],
                        baseline_3nn=point_scores["baseline"]["knn_purity_per_sample"],
                        tmr_3nn=point_scores["tmr_b512"]["knn_purity_per_sample"])
    counts = dict(Counter(labels))
    balanced = len(set(counts.values())) == 1
    scope = "Prior semantic-checkpoint reference" if reference else "VQ paired with the GPT 120k evaluation"
    report = {
        "samples": len(rows), "classes": len(classes), "class_counts": counts,
        "selected": {f: {**states[f], "metrics": metrics[f]} for f in FAMILIES},
        "split": "train+val", "labels_reviewed": False, "independent_test": False,
        "sample_selection_uses_model_features": False, "original_recordings": len(rows),
        "journal_design": {
            "coordinate_files": {f: filename(f, PRIMARY) for f in FAMILIES},
            "context": f"{'Balanced action subset' if balanced else 'Available action subset'}  |  {len(rows)} paired clips  |  {str(next(iter(counts.values())))+' clips per action' if balanced else 'Unequal class sizes; macro metrics'}",
            "subtitles": {"baseline": "Baseline VQ | best reconstruction-FID checkpoint",
                          "tmr_b512": "Previous semantic checkpoint | VQ step 50,000" if reference else "GPT-paired frozen VQ | VQ step 16,000"},
            "marker_size": 7.3 if balanced else 4.7, "legend_columns": 3 if len(classes) == 6 else 4,
            "footer_right": "PCA50 + t-SNE; P=30, E=12, seed=42.",
            "footer_bottom": "Original 512-D macro metrics. Prior semantic checkpoint reference." if reference else "Original 512-D macro metrics. GPT iteration 120,000 does not update this frozen VQ.",
            "scope": scope,
            "scope_note": "\n所给FID 0.3372来自GPT生成评估，不是VQ重建FID。主图使用该GPT实际配套的16000步冻结VQ；50000步权重仅为此前语义checkpoint参考。\n"}}
    save_json(out / "report.json", report)
    grid(out, labels, fits, settings, classes)
    evidence(out, vectors, labels, metrics, classes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/gpt120k_linked_vq_six_actions_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory; earlier diagnostics are preserved")
    parent = PROJECT / "artifacts/dense_train_validation_v1"
    original = read_rows(parent / "samples.csv")
    prior = json.loads((parent / "report.json").read_text())
    audit = json.loads((parent / "selection_audit.json").read_text())
    cfg = json.loads((parent / "config_snapshot.json").read_text())
    assert digest(parent / "samples.csv") == prior["cohort_sha256"]
    identity_path = REPO / "output/hml3d/GPT_amg_b512_full_bf16/vq_identity.json"
    identity = json.loads(identity_path.read_text())
    frozen_path = Path(identity["frozen"])
    assert digest(frozen_path) == digest(Path(identity["source"])) == identity["vq_sha256"]
    log_path = REPO / "output/hml3d/GPT_amg_b512_full_bf16/run.log"
    matches = [line for line in log_path.read_text().splitlines()
               if "2026-10-29 20:42:20,715" in line and "FID. 0.3372" in line and "Iter 120000" in line]
    assert len(matches) == 1
    all_idx = select_indices(original, CLASSES, 260, SEED)
    rows = [original[i] for i in all_idx]
    assert Counter(r["label"] for r in rows) == Counter(dict(zip(CLASSES, [117, 260, 47, 57, 70, 37])))
    assert len(rows) == 588
    states = {
        "baseline": {"name": "net_best_fid", "path": prior["selected"]["baseline"]["path"],
                     "sha256": prior["selected"]["baseline"]["sha256"], "vq_iteration": 31000},
        "tmr_b512": {"name": "frozen_net_best_fid", "path": str(frozen_path),
                     "sha256": identity["vq_sha256"], "vq_iteration": identity["best_iteration"]},
    }
    assert digest(states["baseline"]["path"]) == states["baseline"]["sha256"]
    inputs = {str(path): digest(path) for path in (
        parent / "samples.csv", parent / "report.json", parent / "selection_audit.json",
        parent / "config_snapshot.json", parent / "baseline_features.npz", parent / "tmr_b512_features.npz", identity_path)}
    plan = {
        "user_evaluation_line": matches[0], "evaluation_log": str(log_path),
        "evaluation_is": "GPT generated-motion evaluation at transformer iteration 120000, not a VQ training evaluation",
        "paired_vq_identity": identity, "main_models": states,
        "classes": CLASSES, "selection_seed": SEED,
        "cohorts": {"balanced_222": "37 per class using feature-blind SHA256 ranking",
                    "all_588": "all existing screened candidates in the six requested classes",
                    "previous_50k_reference": "same 222 clips, previous semantic-selected TMR checkpoint; not paired with this GPT"},
        "feature": "unchanged post-quantization temporal mean; 512 dimensions; L2 normalized",
        "primary_projection": {"PCA": 50, "solver": "full", "tSNE": list(PRIMARY), "iterations": 3000,
                               "init": "pca", "metric": "euclidean", "learning_rate": "auto"},
        "balanced_sensitivity_settings": [list(v) for v in SETTINGS],
        "all_588_sensitivity_settings": [[p, 12, 42] for p in (20, 30, 50)],
        "additional_projection": "Cosine t-SNE on original 512-D for balanced222; P30/E12/seed42, 2000 iterations",
        "sampling_stability_seeds": list(range(SEED, SEED+5)),
        "primary_metrics": ["original512D cosine macro silhouette", "original512D recording-excluded macro3NN purity"],
        "extra_metrics": ["macro5NN", "class cosine similarity matrix and within-between gap"],
        "labels_human_reviewed": False, "independent_test": False,
        "new_data_added": False, "new_training": False, "teacher_features_used": False,
        "checkpoint_selection_uses_new_cohort_scores": False,
        "class_selection_informed_by_prior_diagnostics": True,
        "input_sha256": inputs,
    }
    save_json(out / "protocol_before_encoding.json", plan)
    (out / "requested_evaluation_line.txt").write_text(matches[0]+"\n", encoding="utf-8")
    write_rows(out / "samples_588.csv", rows)
    sources = np.array([audit["source_path_by_sample_id"][r["sample_id"]] for r in rows])
    assert len(set(sources)) == len(rows)
    for row in rows:
        assert digest(repo_path(cfg["dataset_root"]) / "new_joint_vecs" / f"{row['motion_id']}.npy") == row["motion_sha256"]
    vectors = {}
    original_ids = np.array([r["sample_id"] for r in original])
    original_labels = np.array([r["label"] for r in original])
    for family in FAMILIES:
        with np.load(parent / f"{family}_features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], original_ids)
            np.testing.assert_array_equal(data["labels"], original_labels)
            assert str(data["checkpoint_sha256"]) == prior["selected"][family]["sha256"]
            vectors[family] = data["post_mean"][all_idx].copy()
    previous_tmr = vectors["tmr_b512"].copy()
    raw = motion_batch(rows, cfg)
    assert raw.shape == (588, 64, 263) and np.isfinite(raw).all()
    torch.set_num_threads(1)
    encoder = FrozenEncoder(cfg, frozen_path).cpu().eval()
    chunks, pre_chunks, tokens = [], [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            pre, post, idx = encoder(torch.from_numpy(raw[start:start+16]))
            chunks.append(post.numpy())
            pre_chunks.append(pre.numpy())
            tokens.append(idx.numpy())
            if start % 160 == 0:
                print(f"Encoding actual GPT-paired frozen VQ: {min(start+16, len(rows))}/588", flush=True)
    vectors["tmr_b512"] = normalize(np.concatenate(chunks))
    np.savez_compressed(out / "frozen_vq_all588_features.npz", post_mean=vectors["tmr_b512"],
                        pre_mean=normalize(np.concatenate(pre_chunks)), tokens=np.concatenate(tokens),
                        labels=np.array([r["label"] for r in rows]), sample_ids=np.array([r["sample_id"] for r in rows]),
                        checkpoint_sha256=states["tmr_b512"]["sha256"])
    del encoder
    # Check that reusing the baseline cache agrees with encoding the same six crops.
    check_idx = [next(i for i, r in enumerate(rows) if r["label"] == label) for label in CLASSES]
    baseline = FrozenEncoder(cfg, states["baseline"]["path"]).cpu().eval()
    with torch.inference_mode():
        fresh = normalize(baseline(torch.from_numpy(raw[check_idx]))[1].numpy())
    np.testing.assert_allclose(fresh, vectors["baseline"][check_idx], atol=2e-6, rtol=2e-5)
    del baseline, raw
    for x in vectors.values():
        assert np.isfinite(x).all()
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    balanced_idx = select_indices(rows, CLASSES, 37, SEED, balanced=True)
    balanced_rows = [rows[i] for i in balanced_idx]
    balanced_vectors = {family: x[balanced_idx] for family, x in vectors.items()}
    reports = {}
    reports["balanced_222"] = cohort(out / "balanced_222", balanced_rows, sources[balanced_idx], balanced_vectors, states, SETTINGS)
    reports["all_588"] = cohort(out / "all_588", rows, sources, vectors, states, [(p, 12, 42) for p in (20, 30, 50)])
    reference_states = copy.deepcopy(states)
    reference_states["tmr_b512"] = {"name": "model_50000", "path": prior["selected"]["tmr_b512"]["path"],
                                    "sha256": prior["selected"]["tmr_b512"]["sha256"], "vq_iteration": 50000}
    reports["previous_50k_reference"] = cohort(out / "previous_50k_reference", balanced_rows, sources[balanced_idx],
        {"baseline": balanced_vectors["baseline"], "tmr_b512": previous_tmr[balanced_idx]}, reference_states, [PRIMARY], reference=True)
    # The alternative projection keeps the same scores, sample list, checkpoint and plotting style.
    alternative = out / "balanced_222_cosine"
    alternative.mkdir()
    write_rows(alternative / "samples.csv", balanced_rows)
    alternative_report = copy.deepcopy(reports["balanced_222"])
    alternative_report["journal_design"]["coordinate_files"] = {}
    alternative_report["journal_design"]["footer_right"] = "Cosine t-SNE on 512-D; P=30, seed=42."
    alternative_report["journal_design"]["footer_bottom"] = "Original 512-D macro metrics. Alternative projection; same frozen VQ and clips."
    for family in FAMILIES:
        xy, kl, trust = fit_cosine(balanced_vectors[family], 30, 42)
        path = alternative / f"{family}_cosine_p30_s42.npz"
        np.savez_compressed(path, coordinates=xy, sample_ids=np.array([r["sample_id"] for r in balanced_rows]),
                            labels=np.array([r["label"] for r in balanced_rows]), kl_divergence=kl,
                            trustworthiness_original_cosine_k10=trust)
        alternative_report["journal_design"]["coordinate_files"][family] = path.name
    save_json(alternative / "report.json", alternative_report)
    stability = []
    for seed in range(SEED, SEED+5):
        idx = select_indices(rows, CLASSES, 37, seed, balanced=True)
        label = np.array([rows[i]["label"] for i in idx])
        stability.append({"sampling_seed": seed, "samples": len(idx),
                          "metrics": {f: score(x[idx], label, sources[idx]) for f, x in vectors.items()}})
    save_json(out / "sampling_stability.json", {"sampling_repeats_can_overlap": True,
              "not_training_repeats_or_independent_tests": True, "results": stability})
    summary = {"protocol": plan, "cohorts": reports}
    save_json(out / "report.json", summary)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert digest(frozen_path) == identity["vq_sha256"]
    save_json(out / "verification.json", {
        "requested_log_line_matched": True, "paired_vq_hash_matches_identity_and_original": True,
        "all_588_raw_motion_hashes_match": True, "588_distinct_original_recordings": True,
        "balanced_222_exactly_37_per_class": Counter(r["label"] for r in balanced_rows) == Counter(dict.fromkeys(CLASSES, 37)),
        "same_samples_and_crops_on_both_sides": True, "baseline_cache_agrees_with_six_fresh_encodings": True,
        "all_features_finite_and_unit_norm": True, "all_input_artifacts_unchanged": True,
        "no_new_data_training_or_teacher_outputs": True,
        "normalization_sha256": {name: digest(repo_path(cfg["normalization_dir"]) / name) for name in ("mean.npy", "std.npy")},
        "new_tmr_model_is_not_a_120000_step_vq_checkpoint": True,
    })
    lines = ["# GPT 120000步日志所配套VQ的六类对比", "",
             "给定FID 0.3372属于GPT生成运动评估。其实际配套VQ为best-FID16000步冻结权重，文件哈希与vq_identity.json及原始权重一致；GPT继续训练不更新该编码器。", "",
             "不新增数据。只使用run、jump、sit、kick、throw、clap六类，全部588段；主图每类37段，共222段。样本按哈希盲选，两模型输入完全一致。标签仍待核验，数据包含训练与验证录制；不是独立确认或严格匹配训练条件的TMR消融。", "",
             "|集合|TMR状态|普通→TMR轮廓系数|普通→TMR 3-NN纯度|", "|---|---|---:|---:|"]
    for name, report in reports.items():
        b, t = [report["selected"][f]["metrics"] for f in FAMILIES]
        state = report["selected"]["tmr_b512"]["name"]
        lines.append(f"|{name}|{state}|{b['macro_silhouette']:.4f}→{t['macro_silhouette']:.4f}|{b['macro_neighbor_purity']:.2%}→{t['macro_neighbor_purity']:.2%}|")
    lines += ["", "主配置PCA50、欧氏t-SNE、P30/E12/seed42、3000次迭代。主图完整9组P×seed敏感性、全部588段三组敏感性、原始512维cosine投影及此前50000步权重参考均保留，不按图形删点或移动类别。", "",
              "各主图下方指标在原始512维计算。original_space_evidence保留同异类余弦相似度与逐类增量；sampling_stability保留五次可能重叠的均衡抽样，不能当作五次训练。条件重抽样区间只重采样固定逐样本分数，不重拟合邻域、不包含训练或标签不确定性。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed scientific analysis: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
