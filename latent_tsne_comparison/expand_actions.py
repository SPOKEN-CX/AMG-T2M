#!/usr/bin/env python3
"""Validation-only expansion and pooling/teacher diagnostics. Never optimizes test outcomes."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, load_config, read_rows, repo_path, save_json, source_id, validate_rows, versions, write_rows
from prepare import captions_for
from features import FrozenEncoder, motion_batch, normalize
from analyze import latent_metrics, run_tsne, scatter

ANNOTATIONS = REPO / "models/Retrival/datasets/annotations/humanml3d/annotations.json"


def recording(annotations, ident):
    path = annotations[ident]["path"]
    return path[2:] if path.startswith("M/") else path


def pool_sequence(sequence, name):
    # B,C,T; each representation is applied identically to both VQ encoders.
    mean = sequence.mean(-1)
    if name == "mean":
        return mean
    if name == "mean_std":
        return torch.cat((mean, sequence.std(-1, unbiased=False)), -1)
    if name == "pyramid2":
        half = sequence.shape[-1] // 2
        if half < 1:
            raise ValueError("Not enough latent timesteps")
        return torch.cat((sequence[..., :half].mean(-1), sequence[..., half:].mean(-1)), -1)
    raise ValueError(f"Unknown pooling: {name}")


def collect(split, cfg, protocol, annotations, *, skip_invalid_captions=False):
    root = repo_path(cfg["dataset_root"])
    patterns = {c["name"]: re.compile(c["pattern"], re.I) for c in protocol["classes"]}
    width = protocol["window_size"]
    groups = defaultdict(list)
    excluded = Counter()
    for ident in (root / f"{split}.txt").read_text().split():
        if ident.startswith("M"):
            continue
        path = root / "new_joint_vecs" / f"{ident}.npy"
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if x.ndim != 2 or x.shape[1] != 263 or len(x) < width or not np.isfinite(x).all():
            excluded["invalid_or_short_motion"] += 1
            continue
        try:
            caps = captions_for(root / "texts" / f"{ident}.txt", len(x), cfg["fps"])
        except ValueError:
            if not skip_invalid_captions:
                raise
            excluded["invalid_caption_metadata"] += 1
            continue
        matches = {label: [c for c in caps if pattern.search(c["text"])] for label, pattern in patterns.items()}
        labels = [label for label, values in matches.items() if values]
        disagreement_note = ""
        if len(labels) > 1 and protocol.get("dominant_caption_rule", False):
            votes = {label: len(matches[label]) for label in labels}
            ranked = sorted(labels, key=lambda label: -votes[label])
            target = ranked[0]
            exclusive = any(not any(patterns[other].search(c["text"]) for other in labels if other != target)
                            for c in matches[target])
            if (votes[target] >= 2 and votes[target]*3 >= len(caps)*2
                    and all(votes[other] <= 1 for other in labels if other != target) and exclusive):
                disagreement_note = "Dominant-caption screening; minor competing mentions: " + json.dumps(votes, sort_keys=True)
                labels = [target]
        if len(labels) != 1:
            excluded["no_label" if not labels else "multiple_actions"] += 1
            continue
        label = labels[0]
        if len(matches[label]) < protocol["minimum_matching_captions"]:
            excluded["insufficient_caption_consensus"] += 1
            continue
        valid = [c for c in matches[label] if c["end"] - c["start"] >= width]
        if not valid:
            excluded["annotated_segment_too_short"] += 1
            continue
        chosen = min(valid, key=lambda c: (c["end"] - c["start"], len(c["text"]), c["text"]))
        lo, hi = chosen["start"], chosen["end"]
        start = lo + (hi - lo - width) // 2
        flags = ["possible_action_sequence"] if any(re.search(r"\b(?:then|afterwards|subsequently)\b", c["text"], re.I) for c in caps) else []
        row = dict(sample_id=f"{ident}_{start:04d}_{start+width:04d}", motion_id=ident,
                   source_id=ident, label=label, start_frame=start, end_frame=start + width,
                   motion_frames=len(x), annotation_start=lo, annotation_end=hi, caption=chosen["text"],
                   captions_json=json.dumps(caps, ensure_ascii=False), caption_votes=len(matches[label]),
                   risk_flags=";".join(flags), review_status="pending", review_note=disagreement_note, motion_sha256=digest(path))
        groups[label].append(row)
    # A recording contributes at most once to the whole cohort, including across labels.
    rows = [r for c in protocol["classes"] for r in groups[c["name"]]]
    rows.sort(key=lambda r: (bool(r["risk_flags"]), -min(2, int(r["caption_votes"])),
              hashlib.sha256(f"{protocol['seed']}:{r['motion_id']}".encode()).hexdigest()))
    seen, unique = set(), defaultdict(list)
    for row in rows:
        key = recording(annotations, row["motion_id"])
        if key in seen:
            excluded["duplicate_recording"] += 1
            continue
        seen.add(key)
        unique[row["label"]].append(row)
    return unique, dict(excluded)


def extract_sequences(cfg, rows, protocol, out, device):
    inputs = motion_batch(rows, cfg)
    result = {}
    for spec in cfg["models"]:
        model = FrozenEncoder(cfg, repo_path(spec["checkpoint"])).to(device).eval()
        captured = []
        handle = model.encoder.register_forward_hook(lambda module, args, output: captured.append(output))
        chunks = defaultdict(list)
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                x = torch.from_numpy(inputs[start:start + 16]).to(device)
                expected_pre, expected_post, indices = model(x)
                pre = captured.pop()
                post = torch.nn.functional.embedding(indices, model.codebook).transpose(1, 2)
                torch.testing.assert_close(pre.mean(-1), expected_pre)
                torch.testing.assert_close(post.mean(-1), expected_post)
                for stage, sequence in (("pre", pre), ("post", post)):
                    for pool in protocol["pools"]:
                        chunks[f"{stage}_{pool}"].append(pool_sequence(sequence, pool).cpu().numpy())
        handle.remove()
        result[spec["name"]] = {name: normalize(np.concatenate(values)) for name, values in chunks.items()}
        np.savez_compressed(out / f"{spec['name']}_features.npz", **result[spec["name"]],
                            sample_ids=np.array([r["sample_id"] for r in rows]))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Validation features extracted: {spec['name']} ({len(rows)} clips, six representations)", flush=True)
    return result, inputs


def teacher_features(cfg, inputs, out, device):
    """Diagnostic teacher only. Never substitutes for the learned VQ representation."""
    base = REPO / "models/Retrival"
    spec = importlib.util.spec_from_file_location("diagnostic_actor", base / "src/model/actor.py")
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    config = json.loads((base / "models/tmr_humanml3d_guoh3dfeats/config.json").read_text())
    kwargs = {k: v for k, v in config["model"]["motion_encoder"].items() if k != "_target_"}
    model = actor.ACTORStyleEncoder(**kwargs)
    weights = base / "models/tmr_humanml3d_guoh3dfeats/last_weights/motion_encoder.pt"
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model.to(device).eval().requires_grad_(False)
    # Recover raw input before applying the teacher's own statistics.
    stats = repo_path(cfg["normalization_dir"])
    raw = inputs * np.load(stats / "std.npy") + np.load(stats / "mean.npy")
    mean = torch.load(base / "stats/humanml3d/guoh3dfeats/mean.pt", map_location="cpu")
    std = torch.load(base / "stats/humanml3d/guoh3dfeats/std.pt", map_location="cpu")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(raw), 16):
            x = ((torch.from_numpy(raw[start:start + 16]).float() - mean) / (std + 1e-12)).to(device)
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
            chunks.append(model({"x": x, "mask": mask})[:, 0].cpu().numpy())
    features = normalize(np.concatenate(chunks))
    np.savez_compressed(out / "teacher_reference.npz", features=features)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return features, {"checkpoint": str(weights), "checkpoint_sha256": digest(weights),
                      "definition": "TMR motion encoder mean token on the identical 64-frame crop; diagnostic reference only, not a theoretical upper bound"}


def macro_metrics(x, rows, annotations, k):
    labels = np.array([r["label"] for r in rows])
    sources = np.array([recording(annotations, r["motion_id"]) for r in rows])
    result = latent_metrics(x, labels, sources, k)
    per_class = {label: {"silhouette": float(result["silhouette_per_sample"][labels == label].mean()),
                         "neighbor_purity": float(result["knn_purity_per_sample"][labels == label].mean()),
                         "samples": int((labels == label).sum())} for label in dict.fromkeys(labels)}
    return {"macro_silhouette": float(np.mean([m["silhouette"] for m in per_class.values()])),
            "macro_neighbor_purity": float(np.mean([m["neighbor_purity"] for m in per_class.values()])),
            "micro_silhouette": result["silhouette_cosine"], "micro_neighbor_purity": result["knn_purity"],
            "per_class": per_class}


def choose_feature(scores, protocol):
    eligible = [name for name, m in scores.items() if m["delta_silhouette"] > 0 and m["delta_purity"] > 0]
    if protocol.get("require_non_degrading_tmr", False):
        primary = scores[protocol["primary_feature"]]["tmr"]
        eligible = [name for name in eligible
                    if scores[name]["tmr"]["macro_silhouette"] >= primary["macro_silhouette"]
                    and scores[name]["tmr"]["macro_neighbor_purity"] >= primary["macro_neighbor_purity"]]
    return max(eligible, key=lambda name: (scores[name]["delta_silhouette"], scores[name]["tmr"]["macro_silhouette"])) if eligible else protocol["primary_feature"]


def make_plot(cfg, rows, features, scores, feature, out):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = np.array([r["label"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=.03, right=.85, bottom=.20, top=.80, wspace=.08)
    for ax, spec, role in zip(axes, cfg["models"], ("baseline", "tmr")):
        x = features[spec["name"]][feature]
        reduced = PCA(n_components=min(50, len(x) - 1, x.shape[1]), svd_solver="full").fit_transform(x)
        xy, kl = run_tsne(reduced, cfg, 30, 42)
        scatter(ax, xy, labels, cfg, size=23)
        ax.set_title(spec["title"], fontsize=13, weight="bold")
        m = scores[feature][role]
        ax.text(.5, -.09, f"Macro silhouette {m['macro_silhouette']:.3f} | 3-NN purity {m['macro_neighbor_purity']:.1%}",
                transform=ax.transAxes, ha="center", fontsize=10)
        np.savez_compressed(out / f"{feature}_{spec['name']}_coordinates.npz", coordinates=xy, labels=labels,
                            sample_ids=np.array([r["sample_id"] for r in rows]), kl_divergence=kl)
    counts = Counter(labels)
    handles = [Line2D([], [], marker="o", ls="", color=c["color"], markersize=6,
               label=f"{c['name']} (n={counts[c['name']]})") for c in cfg["classes"]]
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=9)
    fig.suptitle(f"VALIDATION EXPLORATION — labels pending review\n{feature} | {len(cfg['classes'])} classes, {len(rows)} clips, 64 frames/clip", fontsize=11)
    fig.text(.5, .025, "Original-space metrics, macro-averaged across classes. Independent t-SNE: perplexity=30, seed=42.\n"
             "Feature choice uses validation data; this figure is not an independent test result.", ha="center", fontsize=8, color="#555555")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"validation_{feature}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/expanded_validation_v2")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--protocol", type=Path, default=PROJECT / "expanded_protocol.json")
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "report.json").exists():
        raise FileExistsError("Preserving completed experiment; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    cfg, protocol = load_config(), json.loads(args.protocol.read_text())
    save_json(out / "preregistered_protocol.json", protocol)
    annotations = json.loads(ANNOTATIONS.read_text())
    root = repo_path(cfg["dataset_root"])
    source_sets = {s: {recording(annotations, i) for i in (root / f"{s}.txt").read_text().split()}
                   for s in ("train", "val", "test")}
    val_groups, excluded = collect("val", cfg, protocol, annotations)
    minimum = protocol["minimum_validation_samples_per_class"]
    classes = [c for c in protocol["classes"] if len(val_groups[c["name"]]) >= minimum]
    if len(classes) < 3:
        raise ValueError("Insufficient supported validation classes")
    cfg.update(classes=classes, split="val", purpose="development_diagnostics", window_size=protocol["window_size"], knn_k=protocol["neighbor_k"])
    rows = [r for c in classes for r in val_groups[c["name"]][:protocol["maximum_validation_samples_per_class"]]]
    validate_rows(rows, cfg, require_balance=False)
    write_rows(out / "samples.csv", rows)
    save_json(out / "config_snapshot.json", cfg)
    save_json(out / "source_audit.json", {"annotations_sha256": digest(ANNOTATIONS),
        "source_intersections": {f"{a}/{b}": len(source_sets[a] & source_sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "candidate_counts": {c["name"]: len(val_groups[c["name"]]) for c in protocol["classes"]},
        "selected_counts": dict(Counter(r["label"] for r in rows)), "excluded": excluded,
        "selected_source_paths": {r["motion_id"]: recording(annotations, r["motion_id"]) for r in rows},
        "selected_sources_seen_in_training": sum(recording(annotations, r["motion_id"]) in source_sets["train"] for r in rows)})
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.manual_seed(protocol["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(protocol["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Locked validation cohort: {len(classes)} classes, {len(rows)} source-deduplicated clips", flush=True)
    features, inputs = extract_sequences(cfg, rows, protocol, out, device)
    baseline, tmr = [m["name"] for m in cfg["models"]]
    scores = {}
    for name in features[baseline]:
        a = macro_metrics(features[baseline][name], rows, annotations, protocol["neighbor_k"])
        b = macro_metrics(features[tmr][name], rows, annotations, protocol["neighbor_k"])
        scores[name] = {"baseline": a, "tmr": b, "delta_silhouette": b["macro_silhouette"] - a["macro_silhouette"],
                        "delta_purity": b["macro_neighbor_purity"] - a["macro_neighbor_purity"]}
    chosen = choose_feature(scores, protocol)
    save_json(out / "locked_feature_choice.json", {"feature": chosen, "classes": [c["name"] for c in classes],
              "rule": protocol["selection_rule"], "choice_split": "val", "test_features_consulted": False})
    teacher, teacher_provenance = teacher_features(cfg, inputs, out, device)
    teacher_scores = macro_metrics(teacher, rows, annotations, protocol["neighbor_k"])
    # Inspect availability only after locking the feature choice. No test encoding/scoring.
    test_groups, test_excluded = collect("test", cfg, protocol, annotations)
    old_rows = read_rows(PROJECT / "artifacts/draft/samples.csv")
    old_ids = {r["motion_id"] for r in old_rows}
    old_sources = {recording(annotations, r["motion_id"]) for r in old_rows}
    availability = {}
    for c in classes:
        candidates = test_groups[c["name"]]
        fresh_id = [r for r in candidates if r["motion_id"] not in old_ids]
        fresh_source = [r for r in candidates if recording(annotations, r["motion_id"]) not in old_sources | source_sets["train"] | source_sets["val"]]
        availability[c["name"]] = {"candidate_test_recordings": len(candidates),
            "not_in_previous_180_ids": len(fresh_id), "source_disjoint_from_train_val_previous_plot": len(fresh_source)}
    source_unseen_rows = [r for r in rows if recording(annotations, r["motion_id"]) not in source_sets["train"]]
    unseen_counts = Counter(r["label"] for r in source_unseen_rows)
    unseen_labels = {label for label, n in unseen_counts.items() if n >= 4}
    mask = np.array([r in source_unseen_rows and r["label"] in unseen_labels for r in rows])
    unseen_report = {}
    if len(unseen_labels) >= 3:
        subset = [r for r, yes in zip(rows, mask) if yes]
        unseen_report = {role: macro_metrics(features[name][chosen][mask], subset, annotations, protocol["neighbor_k"])
                         for role, name in (("baseline", baseline), ("tmr", tmr))}
    make_plot(cfg, rows, features, scores, protocol["primary_feature"], out)
    if chosen != protocol["primary_feature"]:
        make_plot(cfg, rows, features, scores, chosen, out)
    from review import generate_review
    generate_review(cfg, out / "samples.csv", out, require_balance=False)
    report = {"status": "validation_exploration_only", "sample_count": len(rows), "classes": [c["name"] for c in classes],
              "counts": dict(Counter(r["label"] for r in rows)), "labels_human_reviewed": False,
              "metric": "original normalized features, cosine silhouette and 3-NN purity; macro average over classes",
              "scores": scores, "chosen_feature": chosen, "teacher_reference": teacher_scores,
              "teacher_provenance": teacher_provenance, "source_unseen_validation_sensitivity": unseen_report,
              "fresh_test_availability_only": availability,
              "recording_disjoint_confirmation_ready": all(v["source_disjoint_from_train_val_previous_plot"] >= protocol["minimum_new_test_recordings_per_class"] for v in availability.values()),
              "test_features_scored": False, "manifest_sha256": digest(out / "samples.csv"), "versions": versions(),
              "checkpoints": {m["name"]: digest(repo_path(m["checkpoint"])) for m in cfg["models"]}}
    save_json(out / "report.json", report)
    table = [{"feature": name, "baseline_silhouette": s["baseline"]["macro_silhouette"], "tmr_silhouette": s["tmr"]["macro_silhouette"],
              "delta_silhouette": s["delta_silhouette"], "baseline_3nn": s["baseline"]["macro_neighbor_purity"],
              "tmr_3nn": s["tmr"]["macro_neighbor_purity"], "delta_3nn": s["delta_purity"]} for name, s in scores.items()]
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines = ["# 扩展动作验证诊断", "", f"候选14类；按样本数量和文本一致性保留{len(classes)}类，共{len(rows)}段验证运动。全部标签仍待核验。",
             "", "|特征|普通 Silhouette|TMR Silhouette|差值|普通3-NN|TMR 3-NN|", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['feature']}|{r['baseline_silhouette']:.4f}|{r['tmr_silhouette']:.4f}|{r['delta_silhouette']:+.4f}|{r['baseline_3nn']:.1%}|{r['tmr_3nn']:.1%}|" for r in table]
    lines += ["", f"验证集选择的特征：{chosen}。", f"独立列出的TMR教师参考：Silhouette={teacher_scores['macro_silhouette']:.4f}，3-NN={teacher_scores['macro_neighbor_purity']:.1%}。教师不是我们的VQ模型，也不是数学上的性能上界。",
              "", "本实验只在val上选择表示，没有计算新的test特征或测试分数。类数量和统计口径变化，不能把新分数与旧6类180段分数直接相减并归因于模型改进。",
              "", "类别保留只由候选数量和文本一致性确定，未根据模型表现挑个别样本。六种表示的全部结果均已报告。验证集上选择后得到的最好差值含选择偏差，不能作为显著提升的最终证明。",
              "", "发现TMR annotations.json含原始录制路径，可以做录制级去重。原官方划分存在同录制不同片段跨split的情况，source_audit.json提供统计；另附去除训练中同录制来源后的验证敏感性分析。",
              "", "下一步确认需要：人工核验标签和动作窗口、足够的未用录制级测试样本、同配置λ=0对照、多训练种子。若独立新样本不足，应补充帧级标注数据或建立新的按录制划分的训练/测试协议。",
              "", "现有普通与TMR模型训练配置不同；本项目不启动新训练，不改动原180段结果。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "chosen_feature": chosen, "primary": scores[protocol["primary_feature"]],
                      "selected": scores[chosen], "teacher": teacher_scores}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
