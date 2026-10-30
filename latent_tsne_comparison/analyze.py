"""Unsupervised t-SNE panels and label-based diagnostics in the original latent space."""
import csv
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_samples
from threadpoolctl import threadpool_limits

from common import digest, read_rows, save_json, validate_rows


def latent_metrics(x, labels, sources, k):
    distances = pairwise_distances(x, metric="cosine")
    distances = np.clip(distances, 0, 2)
    np.fill_diagonal(distances, 0)
    silhouette = silhouette_samples(distances, labels, metric="precomputed")
    neighbor_distances = distances.copy()
    neighbor_distances[sources[:, None] == sources[None, :]] = np.inf
    if k < 1 or (np.isfinite(neighbor_distances).sum(1) < k).any():
        raise ValueError("Not enough independent sources for the requested k")
    neighbors = np.argsort(neighbor_distances, axis=1, kind="stable")[:, :k]
    purity = (labels[neighbors] == labels[:, None]).mean(1)
    return {"silhouette_cosine": float(silhouette.mean()), "knn_purity": float(purity.mean()),
            "silhouette_per_sample": silhouette, "knn_purity_per_sample": purity,
            "neighbors": neighbors}


def paired_bootstrap(a, b, labels, repeats, seed):
    """Paired stratified resampling of fixed per-sample diagnostics, not retraining CIs."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    differences = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        differences.append(float((b[idx] - a[idx]).mean()))
    return {"difference_ours_minus_baseline": float((b - a).mean()),
            "conditional_bootstrap_95_percentile_interval": np.percentile(differences, [2.5, 97.5]).tolist(),
            "repeats": repeats,
            "interpretation": "paired class-stratified resampling of fixed per-sample scores; neighborhoods are not refitted; excludes training/label uncertainty"}


def run_tsne(x, cfg, perplexity, seed):
    if not (0 < perplexity < len(x)):
        raise ValueError(f"t-SNE perplexity {perplexity} must be below sample count {len(x)}")
    kwargs = dict(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                  random_state=seed, metric="euclidean", method="barnes_hut", n_jobs=1)
    # sklearn 1.3 uses n_iter; newer releases use max_iter.
    parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[parameter] = cfg["tsne"]["iterations"]
    model = TSNE(**kwargs)
    # Tiny cohorts incur severe OpenMP overhead on large shared CPU hosts.
    # Apply this after sklearn is loaded so newly loaded thread pools are covered.
    with threadpool_limits(limits=1):
        coords = model.fit_transform(x)
    if not np.isfinite(coords).all():
        raise ValueError("t-SNE produced nonfinite coordinates")
    return coords, float(model.kl_divergence_)


def scatter(ax, xy, labels, cfg, size=21):
    for c in cfg["classes"]:
        points = xy[labels == c["name"]]
        ax.scatter(points[:, 0], points[:, 1], s=size, color=c["color"], alpha=0.84,
                   linewidths=0, rasterized=False)
    ax.set(xticks=[], yticks=[])
    # Preserve t-SNE geometry while keeping every panel the same physical size.
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.09)
    for spine in ax.spines.values():
        spine.set_color("#c4c8ce")
        spine.set_linewidth(0.7)


def plot_pair(cfg, coordinates, labels, metrics, feature, reviewed, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.subplots_adjust(left=0.025, right=0.88, bottom=0.17, top=0.84, wspace=0.075)
    for ax, spec in zip(axes, cfg["models"]):
        scatter(ax, coordinates[spec["name"]], labels, cfg)
        ax.set_title(spec["title"], fontsize=13, weight="bold", pad=12)
        m = metrics[spec["name"]][feature]
        ax.text(0.5, -0.08, f"Silhouette {m['silhouette_cosine']:.3f}   |   {cfg['knn_k']}-NN purity {m['knn_purity']:.1%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
    handles = [Line2D([], [], marker="o", ls="", color=c["color"], label=c["name"], markersize=7)
               for c in cfg["classes"]]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.51), frameon=False,
               labelspacing=1.25)
    feature_label = "Post-quantization" if feature == "post" else "Pre-quantization"
    status = "Human-accepted labels" if reviewed else "CAPTION-SCREENED DRAFT: labels pending human review"
    fig.suptitle(f"{status}  |  {feature_label} temporal-mean features", fontsize=10, color="#555b65", y=0.99)
    n = len(labels) // len(cfg["classes"])
    fig.text(0.5, 0.025,
             f"Same {len(labels)} test clips ({n}/class), {cfg['window_size']} frames/clip. "
             f"Independent t-SNE fits: perplexity={cfg['tsne']['perplexity']}, seed={cfg['tsne']['seed']}.\n"
             "Metrics use the original L2-normalized latent features; panel positions and scales are independent.",
             ha="center", va="bottom", fontsize=8, color="#555b65")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=300, facecolor="white")
    plt.close(fig)


def plot_sensitivity(cfg, fits, labels, feature, reviewed, path):
    seeds, perplexities = cfg["tsne"]["sensitivity_seeds"], cfg["tsne"]["sensitivity_perplexities"]
    fig, axes = plt.subplots(len(perplexities), len(seeds) * 2,
                             figsize=(len(seeds) * 5, len(perplexities) * 2.7), squeeze=False)
    for row, perplexity in enumerate(perplexities):
        for column, seed in enumerate(seeds):
            for j, spec in enumerate(cfg["models"]):
                ax = axes[row, column * 2 + j]
                scatter(ax, fits[(spec["name"], feature, perplexity, seed)], labels, cfg, size=8)
                ax.set_title(f"{'Baseline' if j == 0 else 'TMR-guided'} | seed {seed}", fontsize=9)
                if column == 0 and j == 0:
                    ax.set_ylabel(f"perplexity = {perplexity}", fontsize=10)
    status = "Human-accepted labels" if reviewed else "Caption-screened draft; labels pending review"
    fig.suptitle(f"{feature.capitalize()}-quantization sensitivity | {status}", fontsize=13)
    handles = [Line2D([], [], marker="o", ls="", color=c["color"], label=c["name"], markersize=6)
               for c in cfg["classes"]]
    fig.legend(handles=handles, ncol=len(handles), loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95), h_pad=1.2, w_pad=1.0)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=230, facecolor="white")
    plt.close(fig)


def analyze(cfg, manifest, feature_dir, out, sensitivity=True, resume=False):
    rows = read_rows(manifest)
    counts = validate_rows(rows, cfg)
    requested_perplexities = [cfg["tsne"]["perplexity"]]
    if sensitivity:
        requested_perplexities += cfg["tsne"]["sensitivity_perplexities"]
    if any(p <= 0 or p >= len(rows) for p in requested_perplexities):
        raise ValueError(f"All requested perplexities must be between 0 and N={len(rows)}. "
                         "For a smaller reviewed cohort, predeclare lower values in a new config and re-extract.")
    feature_dir, out = Path(feature_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((feature_dir / "provenance.json").read_text())
    if provenance["manifest_sha256"] != digest(manifest) or provenance["config"] != cfg:
        raise ValueError("Features are stale: configuration or sample manifest changed; re-extract")
    labels = np.array([r["label"] for r in rows])
    sources = np.array([r["source_id"] for r in rows])
    sample_ids = np.array([r["sample_id"] for r in rows])
    metrics, original, reduced, pca_info = {}, {}, {}, {}
    flat_metrics, class_metrics = [], []
    for spec in cfg["models"]:
        name = spec["name"]
        metrics[name] = {}
        with np.load(feature_dir / f"{name}.npz", allow_pickle=False) as data:
            if (str(data["manifest_sha256"]) != digest(manifest)
                    or not np.array_equal(data["sample_ids"], sample_ids)
                    or not np.array_equal(data["labels"], labels)
                    or not np.array_equal(data["source_ids"], sources)):
                raise ValueError("Feature sample IDs, labels or order differ from the manifest")
            for feature in ("pre", "post"):
                x = data[feature].copy()
                if x.shape != (len(rows), cfg["architecture"]["code_dim"]):
                    raise ValueError("Unexpected latent feature dimensions")
                if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
                    raise ValueError("Expected finite L2-normalized features")
                original[(name, feature)] = latent_metrics(x, labels, sources, cfg["knn_k"])
                result = original[(name, feature)]
                metrics[name][feature] = {k: result[k] for k in ("silhouette_cosine", "knn_purity")}
                flat_metrics.append({"model": name, "feature": feature, "samples": len(rows), **metrics[name][feature]})
                for c in cfg["classes"]:
                    mask = labels == c["name"]
                    class_metrics.append({"model": name, "feature": feature, "class": c["name"], "samples": int(mask.sum()),
                                          "silhouette_cosine": float(result["silhouette_per_sample"][mask].mean()),
                                          "knn_purity": float(result["knn_purity_per_sample"][mask].mean())})
                components = min(cfg["tsne"]["pca_components"], x.shape[1], len(x) - 1)
                pca = PCA(n_components=components, svd_solver="full")
                reduced[(name, feature)] = pca.fit_transform(x)
                pca_info[f"{name}/{feature}"] = {"components": components,
                    "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
                # Persist PCA state so no transform is hidden from the experiment record.
                np.savez_compressed(out / f"pca_{name}_{feature}.npz", components=pca.components_, mean=pca.mean_,
                                    explained_variance_ratio=pca.explained_variance_ratio_)
                np.savez_compressed(out / f"per_sample_{name}_{feature}.npz", sample_ids=sample_ids,
                                    silhouette=result["silhouette_per_sample"], knn_purity=result["knn_purity_per_sample"],
                                    neighbor_indices=result["neighbors"])
    comparisons = {}
    base, ours = [m["name"] for m in cfg["models"]]
    for feature in ("pre", "post"):
        comparisons[feature] = {metric: paired_bootstrap(original[(base, feature)][metric + "_per_sample"],
                    original[(ours, feature)][metric + "_per_sample"], labels, cfg["bootstrap_repeats"], cfg["selection_seed"])
                    for metric in ("silhouette", "knn_purity")}
    reviewed = all(r["review_status"] == "accepted" for r in rows)
    report = {"manifest_sha256": digest(manifest), "labels_human_accepted": reviewed, "counts": counts,
              "metric_space": "original L2-normalized latent features; cosine distance; no PCA/t-SNE used for reported scores",
              "knn_k": cfg["knn_k"], "models": metrics, "paired_differences": comparisons, "pca": pca_info}
    save_json(out / "metrics.json", report)
    for filename, records in (("metrics.csv", flat_metrics), ("class_metrics.csv", class_metrics)):
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    tsne = cfg["tsne"]
    main = (tsne["perplexity"], tsne["seed"])
    parameters = {main}
    if sensitivity:
        parameters.update((p, seed) for p in tsne["sensitivity_perplexities"] for seed in tsne["sensitivity_seeds"])
    fits, fitting = {}, []
    feature_hashes = {m["name"]: digest(feature_dir / f"{m['name']}.npz") for m in cfg["models"]}
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    for feature in ("post", "pre"):
        for perplexity, seed in sorted(parameters):
            coordinate_path = out / f"tsne_{feature}_p{perplexity}_s{seed}.npz"
            saved = {"labels": labels, "sample_ids": sample_ids, "manifest_sha256": np.array(digest(manifest))}
            cached, cache_metadata, legacy = {}, {}, False
            if resume and coordinate_path.exists():
                with np.load(coordinate_path, allow_pickle=False) as data:
                    valid = (str(data["manifest_sha256"]) == digest(manifest)
                             and np.array_equal(data["labels"], labels) and np.array_equal(data["sample_ids"], sample_ids))
                    if "config_sha256" in data:
                        valid = valid and str(data["config_sha256"]) == config_hash
                    if "feature_sha256_json" in data:
                        valid = valid and json.loads(str(data["feature_sha256_json"])) == feature_hashes
                    else:
                        # Early files from this same interrupted run have manifest IDs and
                        # the separately validated extraction provenance, but no per-fit hashes.
                        legacy = True
                    if valid:
                        cached = {m["name"]: data[m["name"]].copy() for m in cfg["models"]}
                        if "fit_metadata_json" in data:
                            cache_metadata = json.loads(str(data["fit_metadata_json"]))
            pair_metadata = {}
            for spec in cfg["models"]:
                name = spec["name"]
                if name in cached:
                    xy = cached[name]
                    if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
                        raise ValueError("Invalid cached t-SNE coordinates")
                    kl = cache_metadata.get(name, {}).get("kl_divergence")
                else:
                    print(f"Fitting {name}/{feature}: perplexity={perplexity}, seed={seed}", flush=True)
                    xy, kl = run_tsne(reduced[(name, feature)], cfg, perplexity, seed)
                fits[(name, feature, perplexity, seed)] = xy
                saved[name] = xy
                entry = {"model": name, "feature": feature, "perplexity": perplexity, "seed": seed,
                         "kl_divergence": kl, "reused_cached_coordinates": name in cached}
                if legacy and name in cached:
                    entry["note"] = "Resumed original coordinates from interrupted run; input provenance/IDs checked; per-fit fingerprints added on resume. Original KL was not persisted."
                fitting.append(entry)
                pair_metadata[name] = entry
            saved.update(config_sha256=np.array(config_hash), feature_sha256_json=np.array(json.dumps(feature_hashes)),
                         fit_metadata_json=np.array(json.dumps(pair_metadata)))
            np.savez_compressed(coordinate_path, **saved)
            save_json(out / "tsne_fits.json", fitting)
            print(f"t-SNE {feature}: perplexity={perplexity}, seed={seed} (both models; {'cached' if cached else 'computed'})", flush=True)
        pair = {m["name"]: fits[(m["name"], feature, *main)] for m in cfg["models"]}
        plot_pair(cfg, pair, labels, metrics, feature, reviewed, out / f"comparison_{feature}")
        if sensitivity:
            plot_sensitivity(cfg, fits, labels, feature, reviewed, out / f"sensitivity_{feature}")
    save_json(out / "tsne_fits.json", fitting)
    post = comparisons["post"]
    ds = post["silhouette"]["difference_ours_minus_baseline"]
    dk = post["knn_purity"]["difference_ours_minus_baseline"]
    if ds > 0 and dk > 0:
        interpretation = "两个原始特征空间指标均提高；在当前所选样本和标签条件下，支持 TMR 模型可分性改善。"
    elif ds <= 0 and dk <= 0:
        interpretation = "两个原始特征空间指标均未提高；当前结果不支持 TMR 模型类别可分性改善。"
    else:
        interpretation = "两个指标变化方向不一致；当前结果不能概括为类别可分性全面改善。"
    text = ["# 潜空间特征比较结果", "", f"标签状态：{'已由输入核验清单逐条接受' if reviewed else '自动文本筛选，尚未人工核验；这是探索性草图'}。",
            f"测试集共 {len(rows)} 段，{len(cfg['classes'])} 类，各 {len(rows)//len(cfg['classes'])} 段；每段 {cfg['window_size']} 帧。", "",
            "|模型|特征|余弦 Silhouette ↑|5-NN 同类邻居比例 ↑|", "|---|---|---:|---:|"]
    text += [f"|{m['model']}|{m['feature']}|{m['silhouette_cosine']:.4f}|{m['knn_purity']:.2%}|" for m in flat_metrics]
    text += ["", f"量化后指标差值（TMR−普通）：Silhouette {ds:+.4f}；5-NN {dk * 100:+.2f} 个百分点。", interpretation, "",
             f"指标在原始归一化潜特征上计算，未使用二维 t-SNE 坐标。主图固定 perplexity={cfg['tsne']['perplexity']}、seed={cfg['tsne']['seed']}；稳定性图保留预先指定的全部参数组合。", "",
             "两个模型独立降维，面板坐标、簇间距离、簇面积不能跨面板直接比较。类别标签仅用于着色和计算诊断指标，未输入 PCA/t-SNE。", "",
             "样本是按文本规则筛选的类别平衡子集；全段描述与中心窗口可能不一致。核验页支持播放完整运动、调整64帧窗口、改类及逐条接受/排除。", "",
             "本180段初稿的source_id仅去除HumanML3D镜像重复。后续已在TMR annotations.json找到原始录制路径；扩展验证实验已使用该映射，当前初稿未按录制级协议重建。", "",
             "现有两次训练的 batch、部分损失系数和训练预算不同。本结果是两个现有模型的比较，不能单独归因于 TMR；严格因果消融需另训练同配置 λ=0 模型。", "",
             "metrics.json 中的 bootstrap 区间只对固定特征和标签的逐样本分数重采样，不代表训练随机性、完整总体或标签不确定性的置信区间。", "",
             "主图：comparison_post.png/pdf/svg；量化前补充图：comparison_pre.png/pdf/svg；稳定性图：sensitivity_post/pre.png/pdf。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(text), encoding="utf-8")
    print(interpretation, flush=True)
    return report


def render_saved(cfg, manifest, out):
    """Redraw cached coordinates without re-extracting or re-fitting t-SNE."""
    rows = read_rows(manifest)
    validate_rows(rows, cfg)
    out = Path(out)
    metrics = json.loads((out / "metrics.json").read_text())
    provenance = json.loads((out.parent / "features/provenance.json").read_text())
    if metrics["manifest_sha256"] != digest(manifest) or provenance["config"] != cfg:
        raise ValueError("Cached plot data do not match the manifest/config")
    labels = np.array([r["label"] for r in rows])
    ids = np.array([r["sample_id"] for r in rows])
    fits = {}
    settings = {(cfg["tsne"]["perplexity"], cfg["tsne"]["seed"])}
    settings.update((p, s) for p in cfg["tsne"]["sensitivity_perplexities"] for s in cfg["tsne"]["sensitivity_seeds"])
    for feature in ("post", "pre"):
        for p, seed in settings:
            path = out / f"tsne_{feature}_p{p}_s{seed}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as stored:
                if (str(stored["manifest_sha256"]) != digest(manifest)
                        or not np.array_equal(stored["labels"], labels)
                        or not np.array_equal(stored["sample_ids"], ids)):
                    raise ValueError("Cached t-SNE coordinates have stale sample IDs")
                for spec in cfg["models"]:
                    fits[(spec["name"], feature, p, seed)] = stored[spec["name"]].copy()
        main = {spec["name"]: fits[(spec["name"], feature, cfg["tsne"]["perplexity"], cfg["tsne"]["seed"])]
                for spec in cfg["models"]}
        plot_pair(cfg, main, labels, metrics["models"], feature, metrics["labels_human_accepted"], out / f"comparison_{feature}")
        complete = all((spec["name"], feature, p, seed) in fits for spec in cfg["models"]
                       for p in cfg["tsne"]["sensitivity_perplexities"] for seed in cfg["tsne"]["sensitivity_seeds"])
        if complete:
            plot_sensitivity(cfg, fits, labels, feature, metrics["labels_human_accepted"], out / f"sensitivity_{feature}")
    print("Redrew main and available sensitivity figures from saved coordinates", flush=True)
