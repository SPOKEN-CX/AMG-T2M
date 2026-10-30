#!/usr/bin/env python3
"""Increase real point counts on explicitly labelled train+validation visualization data."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from common import PROJECT, digest, read_rows, repo_path, save_json, write_rows, versions
from expand_actions import ANNOTATIONS, collect, recording, macro_metrics
from features import FrozenEncoder, motion_batch, normalize
from analyze import run_tsne
from render_polished import render


def prepare_dense(parent, out, cfg, protocol, annotations):
    root = repo_path(cfg["dataset_root"])
    candidates, audits = [], {}
    for split in ("val", "train"):
        print(f"Screening {split}: real motion arrays, caption consensus and recording deduplication", flush=True)
        groups, excluded = collect(split, cfg, protocol, annotations, skip_invalid_captions=True)
        audits[split] = {"candidate_counts": {name: len(group) for name, group in groups.items()}, "excluded": excluded}
        for name in [item["name"] for item in cfg["classes"]]:
            for row in groups[name]:
                # Clear single-action captions take priority over additional point count.
                if row["risk_flags"]:
                    audits[split]["excluded"]["action_sequence_in_dense_cohort"] = audits[split]["excluded"].get("action_sequence_in_dense_cohort", 0) + 1
                    continue
                candidates.append((split, row))
    # Keep one original recording across the complete union; val wins any cross-split tie.
    candidates.sort(key=lambda pair: (pair[0] != "val", hashlib.sha256(
        f"{protocol['seed']}:{pair[1]['motion_id']}".encode()).hexdigest()))
    seen, chosen, split_by_id, source_by_id = set(), [], {}, {}
    duplicates = 0
    for split, row in candidates:
        source = recording(annotations, row["motion_id"])
        if source in seen:
            duplicates += 1
            continue
        seen.add(source)
        chosen.append(row)
        split_by_id[row["sample_id"]] = split
        source_by_id[row["sample_id"]] = source
    order = {item["name"]: i for i, item in enumerate(cfg["classes"])}
    chosen.sort(key=lambda row: (order[row["label"]], hashlib.sha256(
        f"{protocol['seed']}:{row['sample_id']}".encode()).hexdigest()))
    counts = Counter(row["label"] for row in chosen)
    if set(counts) != set(order) or min(counts.values()) < 2:
        raise ValueError(f"Insufficient supported actions: {counts}")
    write_rows(out / "samples.csv", chosen)
    audit = {"selection_splits": ["train", "val"], "test_motions_used": False,
             "counts": dict(counts), "split_counts": dict(Counter(split_by_id.values())),
             "per_class_split_counts": {name: dict(Counter(split_by_id[row["sample_id"]]
                  for row in chosen if row["label"] == name)) for name in counts},
             "unique_recordings": len(seen), "cross_split_recordings_removed": duplicates,
             "source_path_by_sample_id": source_by_id, "split_by_sample_id": split_by_id,
             "screening": audits, "annotations_sha256": digest(ANNOTATIONS),
             "manifest_sha256": digest(out / "samples.csv")}
    save_json(out / "selection_audit.json", audit)
    return chosen, audit


def validate_dense(rows, cfg, audit):
    root = repo_path(cfg["dataset_root"])
    split_ids = {split: set((root / f"{split}.txt").read_text().split()) for split in ("train", "val")}
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(rows) or len(set(audit["source_path_by_sample_id"].values())) != len(rows):
        raise ValueError("Duplicate sample or original recording")
    for row in rows:
        split = audit["split_by_sample_id"][row["sample_id"]]
        if row["motion_id"] not in split_ids[split] or row["motion_id"].startswith("M"):
            raise ValueError("Unexpected motion split or mirrored duplicate")
        start, end, lo, hi = (int(row[key]) for key in ("start_frame", "end_frame", "annotation_start", "annotation_end"))
        if not 0 <= lo <= start < end <= hi <= int(row["motion_frames"]) or end-start != 64:
            raise ValueError("Invalid fixed motion crop")
        if int(row["caption_votes"]) < 2 or row["risk_flags"] or row["review_status"] != "pending":
            raise ValueError("Unexpected screening or review status")
        if digest(root / "new_joint_vecs" / f"{row['motion_id']}.npy") != row["motion_sha256"]:
            raise ValueError("Motion changed after screening")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PROJECT / "artifacts/expanded_validation_v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/dense_train_validation_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    if (out / "report.json").exists():
        raise FileExistsError("Preserving completed dense comparison; use a new output directory")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Use --resume to continue an incomplete run")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((parent / "config_snapshot.json").read_text())
    teacher_free_report = json.loads((parent / "checkpoint_probe/report.json").read_text())
    selected = teacher_free_report["selected"]
    cfg.update(split="train+val", purpose="training_inclusive_visualization", samples_per_class=None)
    for spec in cfg["models"]:
        spec["checkpoint"] = selected[spec["name"]]["path"]
        if digest(spec["checkpoint"]) != selected[spec["name"]]["sha256"]:
            raise ValueError("Previously selected checkpoint changed")
    protocol = json.loads((PROJECT / "expanded_protocol.json").read_text())
    plan = {"purpose": "Increase actual sample count for descriptive visualization",
            "selection_splits": ["train", "val"], "test_motions_used": False,
            "per_class_cap": None, "minimum_caption_votes": 2,
            "screening_patterns": protocol["classes"], "classes": [c["name"] for c in cfg["classes"]],
            "selection": "One pure target action across captions; exclude sequencing flags; one 64-frame center crop per original recording; no feature-based sample filtering",
            "selected_checkpoint_sha256": {name: item["sha256"] for name, item in selected.items()},
            "feature": "L2-normalized temporal mean of post-quantization embeddings, 512D",
            "tsne": cfg["tsne"], "checkpoint_selection": "Locked to the earlier validation probe; no new model selection",
            "independent_test": False}
    protocol_path = out / "protocol_before_encoding.json"
    if args.resume and protocol_path.exists() and json.loads(protocol_path.read_text()) != plan:
        raise ValueError("Cannot resume after changing the locked protocol")
    save_json(protocol_path, plan)
    save_json(out / "config_snapshot.json", cfg)
    annotations = json.loads(ANNOTATIONS.read_text())
    if args.resume and (out / "samples.csv").exists():
        rows = read_rows(out / "samples.csv")
        audit = json.loads((out / "selection_audit.json").read_text())
        if digest(out / "samples.csv") != audit["manifest_sha256"]:
            raise ValueError("Selection manifest changed")
    else:
        rows, audit = prepare_dense(parent, out, cfg, protocol, annotations)
    validate_dense(rows, cfg, audit)
    print(f"Locked {len(rows)} clips, {len(audit['counts'])} actions, {audit['unique_recordings']} distinct recordings; {audit['counts']}", flush=True)
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    inputs = motion_batch(rows, cfg)
    coordinates, evaluated = {}, {}
    for spec in cfg["models"]:
        name = spec["name"]
        path = out / f"{name}_features.npz"
        checkpoint_hash = selected[name]["sha256"]
        if args.resume and path.exists():
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["sample_ids"], ids)
                if str(data["checkpoint_sha256"]) != checkpoint_hash or str(data["manifest_sha256"]) != audit["manifest_sha256"]:
                    raise ValueError("Stale cached features")
                features = data["post_mean"].copy()
        else:
            model = FrozenEncoder(cfg, spec["checkpoint"]).to(device).eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(rows), 16):
                    chunks.append(model(torch.from_numpy(inputs[start:start+16]).to(device))[1].cpu().numpy())
                    if start % 512 == 0:
                        print(f"Encoding {name}: {min(start+16, len(rows))}/{len(rows)}", flush=True)
            features = normalize(np.concatenate(chunks))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            np.savez_compressed(path, post_mean=features, sample_ids=ids, labels=labels,
                                checkpoint_sha256=checkpoint_hash, manifest_sha256=audit["manifest_sha256"])
        metrics = macro_metrics(features, rows, annotations, 3)
        evaluated[name] = {**selected[name], "metrics": metrics}
        path = out / f"{name}_coordinates.npz"
        if args.resume and path.exists():
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["sample_ids"], ids)
                np.testing.assert_array_equal(data["labels"], labels)
                if str(data["features_sha256"]) != digest(out / f"{name}_features.npz"):
                    raise ValueError("Stale cached t-SNE coordinates")
                xy, kl = data["coordinates"].copy(), float(data["kl_divergence"])
        else:
            print(f"Fitting PCA50 + t-SNE for {name} on {len(rows)} real samples (P=30, seed=42)", flush=True)
            pca = PCA(n_components=50, svd_solver="full")
            reduced = pca.fit_transform(features)
            xy, kl = run_tsne(reduced, cfg, 30, 42)
            np.savez_compressed(path, coordinates=xy, labels=labels, sample_ids=ids, kl_divergence=kl,
                                features_sha256=digest(out / f"{name}_features.npz"))
            np.savez_compressed(out / f"{name}_pca.npz", components=pca.components_, mean=pca.mean_,
                                explained_variance_ratio=pca.explained_variance_ratio_)
        if xy.shape != (len(rows), 2) or not np.isfinite(xy).all() or not np.isfinite(kl):
            raise ValueError("Invalid dimensionality reduction")
        coordinates[name] = xy
        print(f"Completed {name}: macro silhouette={metrics['macro_silhouette']:.4f}, 3-NN={metrics['macro_neighbor_purity']:.2%}, KL={kl:.4f}", flush=True)
    report = {"samples": len(rows), "classes": len(cfg["classes"]), "split": "train+val", "selected": evaluated,
              "class_counts": audit["counts"], "split_counts": audit["split_counts"],
              "independent_test": False, "test_motions_used": False, "labels_reviewed": False,
              "cohort_sha256": audit["manifest_sha256"], "versions": versions(), "device": device,
              "display": {"data_label": "HumanML3D train + validation",
                          "note": "Training-inclusive visualization; one clip per original recording; caption labels await review.",
                          "marker_size": max(12, 72 * (108 / len(rows)) ** .55)}}
    for annotated in (False, True):
        rendered = render(out, [c["name"] for c in cfg["classes"]], rows, report, coordinates, annotated=annotated)
        if rendered != {name: len(rows) for name in evaluated}:
            raise ValueError("Not all real samples were plotted")
    save_json(out / "report.json", report)
    lines = ["# 增加真实样本的潜空间可视化", "", f"本轮两侧各{len(rows)}个真实运动片段，10类；每段64帧，每份原始录制仅取一个片段。", "",
             "取消上一版每类24段上限，使用HumanML3D训练集和验证集；不使用新的测试运动、不选新的权重，也不进行训练。",
             "不是独立测试图。动作标签来自文本筛选，至少两条描述命中同一目标类别，排除序列连接词，标签仍待人工核验。", "",
             "|动作|片段数|", "|---|---:|"]
    lines += [f"|{name}|{n}|" for name, n in audit["counts"].items()]
    lines += ["", "|模型|权重|原始特征宏平均Silhouette|宏平均3-NN纯度|", "|---|---|---:|---:|"]
    lines += [f"|{name}|{item['name']}|{item['metrics']['macro_silhouette']:.4f}|{item['metrics']['macro_neighbor_purity']:.2%}|" for name, item in evaluated.items()]
    lines += ["", "两侧使用相同ID、相同裁剪和原训练归一化统计；主特征仍为量化后时间均值。PCA50→t-SNE，perplexity30、seed42、2000次迭代，两边独立拟合，标签只用于着色。",
              "不存在按类移动、重复抖动、人工合成点或将教师特征替代学生特征的处理。增加点数提升图的密度，分离程度仍由真实模型决定。",
              "", "PNG600dpi，PDF和SVG可缩放编辑；preview为便于查看的小图。annotated版本增加原特征空间的宏平均指标。",
              "与先前108段验证样本的指标不能直接比较；本轮包含训练运动，不能据此声称泛化提升或严格TMR因果消融。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(out / "verification.json", {"status": "complete", "samples_per_panel": len(rows),
        "unique_recordings": audit["unique_recordings"], "same_ids_and_labels": True,
        "test_motions_used": False, "new_training_started": False, "teacher_features_used": False,
        "checkpoint_sha256": plan["selected_checkpoint_sha256"], "manifest_sha256": audit["manifest_sha256"],
        "output_sha256": {path.name: digest(path) for path in out.iterdir() if path.suffix in (".png", ".pdf", ".svg", ".npz")}})
    print(f"Dense comparison ready: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
