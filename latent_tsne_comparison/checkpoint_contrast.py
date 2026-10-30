#!/usr/bin/env python3
"""Explicit weaker/stronger checkpoint illustration, with a best-checkpoint control."""
import argparse
import copy
import inspect
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits

from common import PROJECT, digest, read_rows, save_json, write_rows
from features import FrozenEncoder, motion_batch, normalize
from optimize_display import score
from render_polished import PALETTE


SETTINGS = ((50, 24, 42), (30, 12, 42), (50, 12, 42), (50, 24, 0))
MODEL_KEYS = ("baseline_weaker", "baseline_best", "tmr_best")


def projection(features, setting):
    p, exaggeration, seed = setting
    kwargs = dict(n_components=2, perplexity=p, early_exaggeration=exaggeration,
                  init="pca", learning_rate="auto", random_state=seed,
                  metric="euclidean", method="barnes_hut", n_jobs=1)
    parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[parameter] = 3000
    model = TSNE(**kwargs)
    xy = model.fit_transform(features)
    if not np.isfinite(xy).all() or not np.isfinite(model.kl_divergence_):
        raise ValueError("Nonfinite t-SNE fit")
    return xy, float(model.kl_divergence_)


def sensitivity(out, labels, classes, fits):
    fig, axes = plt.subplots(len(SETTINGS), 3, figsize=(13.5, 11))
    for i, setting in enumerate(SETTINGS):
        for j, key in enumerate(MODEL_KEYS):
            ax, xy = axes[i, j], fits[key][setting]
            for label in classes:
                points = xy[labels == label]
                ax.scatter(points[:, 0], points[:, 1], s=5, color=PALETTE[label], alpha=.85, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_color("#D0D7DE")
            names = {"baseline_weaker": "Baseline: lower-scoring checkpoint", "baseline_best": "Baseline: best checkpoint", "tmr_best": "TMR: best checkpoint"}
            ax.set_title(f"{names[key]}\nP={setting[0]}, exaggeration={setting[1]}, seed={setting[2]}", fontsize=8.5)
    fig.suptitle("All parameter settings | same 1,198 clips | primary: P50, exaggeration24, seed42", fontsize=12)
    fig.tight_layout(rect=(0, .01, 1, .95))
    fig.savefig(out / "parameter_comparison.png", dpi=200)
    fig.savefig(out / "parameter_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PROJECT / "artifacts/comparison_1200_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/checkpoint_contrast_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    if (out / "report.json").exists():
        raise FileExistsError("Completed contrast preserved; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    validation_path = PROJECT / "artifacts/expanded_validation_v1/checkpoint_probe/report.json"
    validation = json.loads(validation_path.read_text())
    candidates = validation["all_candidates"]
    weakest = min(candidates["baseline"], key=lambda x: (x["metrics"]["macro_silhouette"], x["metrics"]["macro_neighbor_purity"], x["name"] != "net_last"))
    strongest = {family: max(items, key=lambda x: (x["metrics"]["macro_silhouette"], x["metrics"]["macro_neighbor_purity"]))
                 for family, items in candidates.items()}
    selected = {"baseline_weaker": weakest, "baseline_best": strongest["baseline"], "tmr_best": strongest["tmr_b512"]}
    cfg = json.loads((PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json").read_text())
    rows = read_rows(parent / "samples.csv")
    prior = json.loads((parent / "report.json").read_text())
    audit = json.loads((PROJECT / "artifacts/dense_train_validation_v1/selection_audit.json").read_text())
    ids = np.array([r["sample_id"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    classes = list(prior["class_counts"])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    if len(rows) != 1198 or len(set(ids)) != len(rows) or len(set(sources)) != len(rows):
        raise ValueError("The fixed independent cohort differs")
    input_hashes = {str(parent / n): digest(parent / n) for n in ("samples.csv", "report.json", "baseline_features.npz", "tmr_b512_features.npz")}
    input_hashes[str(validation_path)] = digest(validation_path)
    plan = {"purpose": "User-requested weaker-baseline / stronger-TMR checkpoint illustration",
            "selection_on": "Earlier 108-clip validation diagnostic, before this 1198-clip projection",
            "baseline_selection": "minimum macro cosine silhouette, tie-break minimum macro3NN, prefer net_last if equal",
            "tmr_selection": "maximum macro cosine silhouette, tie-break maximum macro3NN",
            "all_validation_candidates": candidates,
            "selected_checkpoints": selected, "input_sha256": input_hashes,
            "samples": len(rows), "feature": "Unchanged post-quantization temporal mean, L2 normalized512D",
            "projection": {"pca_components": 50, "pca_solver": "full", "iterations": 3000,
                           "metric": "euclidean", "initialization": "pca", "learning_rate": "auto",
                           "primary": list(SETTINGS[0]), "all_settings": [list(v) for v in SETTINGS]},
            "scope": "Illustrative checkpoint contrast; separately retain best-checkpoint control; not a matched causal ablation",
            "new_training": False, "new_test_evaluation": False, "teacher_features_used": False}
    plan_path = out / "protocol_before_projection.json"
    if plan_path.exists() and (not args.resume or json.loads(plan_path.read_text()) != plan):
        raise ValueError("Use --resume with the identical locked protocol")
    save_json(plan_path, plan)
    write_rows(out / "samples.csv", rows)
    features, metrics, fits, records = {}, {}, {}, []
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    for key in MODEL_KEYS:
        state = selected[key]
        if digest(state["path"]) != state["sha256"]:
            raise ValueError("Checkpoint changed since validation diagnostic")
        path = out / f"{key}_features.npz"
        if args.resume and path.exists():
            with np.load(path) as stored:
                np.testing.assert_array_equal(stored["sample_ids"], ids)
                if str(stored["checkpoint_sha256"]) != state["sha256"]:
                    raise ValueError("Stale feature cache")
                x = stored["post_mean"].copy()
        elif key == "baseline_weaker":
            print(f"Encoding lower-scoring baseline: {state['name']} on CPU", flush=True)
            batch = motion_batch(rows, cfg)
            model = FrozenEncoder(cfg, state["path"]).cpu().eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(rows), 16):
                    chunks.append(model(torch.from_numpy(batch[start:start+16]))[1].numpy())
            x = normalize(np.concatenate(chunks))
            del model, batch
        else:
            family = "baseline" if key == "baseline_best" else "tmr_b512"
            with np.load(parent / f"{family}_features.npz") as stored:
                np.testing.assert_array_equal(stored["sample_ids"], ids)
                if str(stored["checkpoint_sha256"]) != state["sha256"]:
                    raise ValueError("Selected best checkpoint differs from parent")
                x = stored["post_mean"].copy()
        features[key] = x
        np.savez_compressed(path, post_mean=x, labels=labels, sample_ids=ids, checkpoint_sha256=state["sha256"])
        metrics[key] = score(x, labels, sources)
        print(f"{key}/{state['name']}: silhouette={metrics[key]['macro_silhouette']:.4f}, 3-NN={metrics[key]['macro_neighbor_purity']:.2%}", flush=True)
        pca = PCA(n_components=50, svd_solver="full")
        reduced = pca.fit_transform(x)
        np.savez_compressed(out / f"{key}_pca.npz", components=pca.components_, mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_)
        fits[key] = {}
        for setting in SETTINGS:
            p, exaggeration, seed = setting
            filename = out / f"{key}_p{p}_e{exaggeration}_s{seed}.npz"
            if args.resume and filename.exists():
                with np.load(filename) as stored:
                    np.testing.assert_array_equal(stored["sample_ids"], ids)
                    if str(stored["features_sha256"]) != digest(path):
                        raise ValueError("Stale coordinate cache")
                    xy, kl = stored["coordinates"].copy(), float(stored["kl_divergence"])
            else:
                xy, kl = projection(reduced, setting)
                np.savez_compressed(filename, coordinates=xy, labels=labels, sample_ids=ids, kl_divergence=kl,
                                    features_sha256=digest(path))
            fits[key][setting] = xy
            records.append({"model": key, "checkpoint": state["name"], "perplexity": p,
                            "early_exaggeration": exaggeration, "seed": seed, "kl_divergence": kl,
                            "trustworthiness_original_cosine_k10": float(trustworthiness(x, xy, metric="cosine", n_neighbors=10))})
            save_json(out / "fits.json", records)
            print(f"{key}: P{p}, E{exaggeration}, seed{seed}, KL={kl:.4f}", flush=True)
    scope_note = "\n本图按用户要求对照较弱普通VQ权重与较强TMR权重，选择依据为先前108段验证集的宏平均可分性。另保留各自最佳权重参考图；选定状态的差异不能解释为严格单变量TMR消融。\n"
    for folder, left, context in (("selected_contrast", "baseline_weaker", "Selected checkpoint contrast"),
                                  ("best_checkpoint_reference", "baseline_best", "Best-checkpoint reference")):
        path = out / folder
        path.mkdir(exist_ok=True)
        write_rows(path / "samples.csv", rows)
        model_map = {"baseline": left, "tmr_b512": "tmr_best"}
        selected_report, coordinate_files = {}, {}
        for family, key in model_map.items():
            selected_report[family] = {**selected[key], "metrics": metrics[key]}
            filename = f"{family}_p50_e24_s42.npz"
            shutil.copy2(out / f"{key}_p50_e24_s42.npz", path / filename)
            coordinate_files[family] = filename
            shutil.copy2(out / f"{key}_features.npz", path / f"{family}_features.npz")
        report = {"samples": len(rows), "class_counts": prior["class_counts"], "classes": len(classes),
                  "selected": selected_report, "split": "train+val", "labels_reviewed": False,
                  "checkpoint_selection": plan["selection_on"], "independent_test": False,
                  "journal_design": {"coordinate_files": coordinate_files,
                      "context": f"{context}  |  {len(rows):,} paired clips  |  {len(classes)} actions",
                      "subtitles": {"baseline": f"{selected[left]['name']} | {'lower' if left == 'baseline_weaker' else 'higher'} validation separability",
                                    "tmr_b512": "model_50000 | higher validation separability"},
                      "footer_right": "PCA50 + t-SNE; P=50, exaggeration=24, seed=42.",
                      "footer_bottom": "Selected checkpoint states; original-space macro metrics. Full parameter comparison retained.",
                      "scope": plan["scope"] if left == "baseline_weaker" else "Both families selected for maximum earlier validation separability",
                      "scope_note": scope_note if left == "baseline_weaker" else "\n参考图保留两边各自较优权重，使用与所选状态对比图相同样本和降维设置。\n"}}
        save_json(path / "report.json", report)
    sensitivity(out, labels, classes, fits)
    report = {"models": {key: {**selected[key], "metrics": metrics[key]} for key in MODEL_KEYS},
              "samples": len(rows), "class_counts": prior["class_counts"], "source_recordings": len(set(sources)),
              "primary_projection": plan["projection"], "validation_selection": str(validation_path),
              "illustrative_checkpoint_contrast": True, "best_checkpoint_control_included": True,
              "new_training": False, "new_test_evaluation": False, "teacher_features_used": False}
    save_json(out / "report.json", report)
    lines = ["# 所选checkpoint的探索性对比", "", "按用户要求选取验证可分性较低的普通VQ权重和较高的TMR权重。选择依据来自已有108段验证诊断，未按新1198点图形选择样本、删点或移动点。", "",
             "普通较低：net_last（与net_best_div的先前验证分数并列）。TMR较高：model_50000。另保留普通net_best_fid与TMR model_50000的参考图。", "",
             "|模型状态|权重|1198段宏平均Silhouette|1198段宏平均3-NN|", "|---|---|---:|---:|"]
    lines += [f"|{key}|{selected[key]['name']}|{metrics[key]['macro_silhouette']:.4f}|{metrics[key]['macro_neighbor_purity']:.2%}|" for key in MODEL_KEYS]
    lines += ["", "新主图PCA50→欧氏t-SNE，perplexity50、early_exaggeration24、seed42、迭代3000。全部模型使用相同设置。另保留P30/E12/seed42、P50/E12/seed42及P50/E24/seed0。所有结果见parameter_comparison。", "",
              "1198段运动来自训练加验证，标签待核验。选定状态对比是描述性示例，反映checkpoint选择与训练过程差异；不能当作双方最佳性能的比较，也不是严格TMR因果消融。未进行新训练或独立测试。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    if input_hashes != {path: digest(path) for path in input_hashes}:
        raise AssertionError("An original artifact changed")
    print(f"Checkpoint contrast completed: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
