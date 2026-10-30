#!/usr/bin/env python3
"""Compare the specified 16k best-FID TMR-VQ against net_last on 700 caption-screened clips."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, read_rows, repo_path, save_json, write_rows
from compare_gpt_linked_vq import FAMILIES, SETTINGS, cohort
from features import FrozenEncoder, motion_batch, normalize
from optimize_display import fit_cosine
from pilot_actions700 import ranked_indices


CLASSES = ["run", "jump", "sit", "kick", "throw", "dance", "wave"]
COUNTS = {"run": 112, "jump": 111, "sit": 81, "kick": 75, "throw": 98, "dance": 112, "wave": 111}


def review_page(rows, cfg, out):
    from utils.motion_process import recover_from_ric
    samples = []
    root = repo_path(cfg["dataset_root"]) / "new_joint_vecs"
    with torch.inference_mode():
        for row in rows:
            raw = np.load(root / f"{row['motion_id']}.npy", allow_pickle=False).astype(np.float32)
            joints = recover_from_ric(torch.from_numpy(raw), 22).numpy()
            assert np.isfinite(joints).all()
            samples.append({"row": row, "joints": np.round(joints, 3).tolist()})
    payload = {"manifest_sha256": digest(out / "samples.csv"), "fps": 20, "window_size": 64,
               "classes": [{"name": label} for label in CLASSES], "samples": samples}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = (PROJECT / "review_template.html").read_text()
    (out / "review.html").write_text(template.replace("__REVIEW_DATA__", serialized), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/bestfid16k_vs_weaker_700_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Preserve previous results; choose a new output directory")
    parent = PROJECT / "artifacts/action_search_700_merged_v1"
    candidates = read_rows(parent / "candidates.csv")
    audit = json.loads((parent / "selection_audit.json").read_text())
    pilot = json.loads((parent / "pilot/report.json").read_text())
    indices = np.concatenate([ranked_indices(candidates, label, audit)[:COUNTS[label]] for label in CLASSES])
    rows = [candidates[i] for i in indices]
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    assert len(rows) == len(set(ids)) == len(set(sources)) == sum(COUNTS.values()) == 700
    assert Counter(labels) == Counter(COUNTS)
    assert max(COUNTS.values())/min(COUNTS.values()) <= 1.5
    cfg = json.loads((PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json").read_text())
    base_checkpoint = REPO / "output/hml3d/VQ_base/net_last.pth"
    tmr_checkpoint = REPO / "output/hml3d/VQ_amg_b512_k50_100k/net_best_fid.pth"
    summary_path = REPO / "output/hml3d/VQ_amg_b512_k50_100k/result_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["best_iteration"] == 16000
    assert min(summary["validation_fid"], key=lambda v: v["fid"])["iteration"] == 16000
    log_path = REPO / "output/hml3d/VQ_amg_b512_k50_100k/run.log"
    lines = [line for line in log_path.read_text().splitlines()
             if "2026-10-29 00:51:50,884" in line and "Eva. Iter 16000" in line and "FID. 0.0664" in line]
    assert len(lines) == 1
    states = {
        "baseline": {"name": "net_last", "path": str(base_checkpoint), "sha256": digest(base_checkpoint),
                     "vq_iteration": 300000, "logged_validation_fid": .1229},
        "tmr_b512": {"name": "net_best_fid", "path": str(tmr_checkpoint), "sha256": digest(tmr_checkpoint),
                     "vq_iteration": 16000, "logged_validation_fid": .0664, "validation_fid": summary["best_fid"]},
    }
    assert states["baseline"]["sha256"] == pilot["plan"]["models"]["baseline"]["sha256"]
    assert states["tmr_b512"]["sha256"] == pilot["plan"]["models"]["tmr_b512"]["sha256"]
    tiers = dict(Counter(audit["tier_by_sample_id"][ident] for ident in ids))
    inputs = {str(path): digest(path) for path in (
        parent / "candidates.csv", parent / "selection_audit.json", parent / "pilot/report.json",
        summary_path, log_path, PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json")}
    plan = {
        "latest_user_requested_vq_evaluation": lines[0], "states": states,
        "classes": CLASSES, "class_counts": COUNTS, "samples": 700,
        "count_ratio_max_to_min": max(COUNTS.values())/min(COUNTS.values()),
        "shares": {label: n/700 for label, n in COUNTS.items()},
        "slightly_upweighted_actions": ["run", "dance"],
        "allocation_rationale": "Keep all available sit81/kick75/throw98; remaining four classes111-112 each; run and dance receive one extra clip",
        "selection": "caption-evidence tier then fixed SHA256(seed:sample_id); no individual model-score filtering",
        "selection_seed": 20261030, "caption_evidence_tiers": tiers,
        "class_selection_scope": "Post-hoc exploration based on earlier432-clip pilot and semantic specificity; full census and all original pilot compositions retained",
        "design_refinement_from_pilot": "Prefer concrete action categories over broad arm_raise; include positive-gain kick despite75 available; relax draft max/min1.35 to1.5 before new700 scoring to reach700 without duplication",
        "pilot_class_silhouette_gains": {label: pilot["class_gains"][label] for label in CLASSES},
        "split": "train+val", "labels_human_reviewed": False, "independent_test": False,
        "window_size": 64, "fps": 20, "feature": "L2-normalized512-D post-quantization temporal mean",
        "primary_projection": {"pca": 50, "solver": "full", "metric": "euclidean", "perplexity": 30,
                               "early_exaggeration": 12, "seed": 42, "init": "pca", "learning_rate": "auto", "max_iter": 3000},
        "all_projection_settings": [list(v) for v in SETTINGS],
        "teacher_outputs_used": False, "new_training": False, "new_external_dataset": False,
        "input_sha256": inputs,
    }
    save_json(out / "protocol_before_encoding.json", plan)
    write_rows(out / "samples.csv", rows)
    (out / "requested_evaluation_line.txt").write_text(lines[0]+"\n", encoding="utf-8")
    save_json(out / "selection_audit.json", {
        "counts": COUNTS, "source_path_by_sample_id": {ident: audit["source_path_by_sample_id"][ident] for ident in ids},
        "split_by_sample_id": {ident: audit["split_by_sample_id"][ident] for ident in ids},
        "tier_by_sample_id": {ident: audit["tier_by_sample_id"][ident] for ident in ids},
        "tier_counts": tiers, "manifest_sha256": digest(out / "samples.csv"),
        "labels_human_reviewed": False, "test_motions_used": False})
    for row in rows:
        assert digest(repo_path(cfg["dataset_root"]) / "new_joint_vecs" / f"{row['motion_id']}.npy") == row["motion_sha256"]
        assert int(row["end_frame"])-int(row["start_frame"]) == 64 and not row["risk_flags"]
    raw = motion_batch(rows, cfg)
    assert raw.shape == (700, 64, 263) and np.isfinite(raw).all()
    torch.set_num_threads(1)
    vectors = {}
    for family, state in states.items():
        model = FrozenEncoder(cfg, state["path"]).cpu().eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                chunks.append(model(torch.from_numpy(raw[start:start+16]))[1].numpy())
                if start % 160 == 0:
                    print(f"Encoding {family}: {min(start+16, len(rows))}/700", flush=True)
        vectors[family] = normalize(np.concatenate(chunks))
        del model
    report = cohort(out, rows, sources, vectors, states, SETTINGS, classes=CLASSES)
    report["selection_protocol"] = plan
    report["split_counts"] = dict(Counter(audit["split_by_sample_id"][ident] for ident in ids))
    report["journal_design"].update({
        "context": "Selected checkpoint contrast  |  700 paired clips  |  7 actions",
        "subtitles": {"baseline": "net_last | VQ step 300,000 | validation FID 0.1229",
                      "tmr_b512": "net_best_fid | VQ step 16,000 | validation FID 0.0664"},
        "marker_size": 4.2, "legend_columns": 4,
        "footer_bottom": "Selected checkpoint states; original 512-D macro metrics. Post-hoc action subset.",
        "scope": "Explicit weaker-baseline vs specified best-FID TMR-VQ; post-hoc training-inclusive action subset",
        "scope_note": "\n本轮明确使用2026-10-29 00:51:50的VQ16000步、FID0.0664权重，不以GPT生成评估代替VQ重建指标。普通侧为较弱net_last。动作及配额在700段编码前固定，类别依据此前开发诊断选择；标签待核验。\n"})
    save_json(out / "report.json", report)
    alternative = out / "cosine_supplement"
    alternative.mkdir()
    write_rows(alternative / "samples.csv", rows)
    alt = copy.deepcopy(report)
    alt["journal_design"]["coordinate_files"] = {}
    alt["journal_design"]["footer_right"] = "Cosine t-SNE on 512-D; P=30, seed=42."
    for family in FAMILIES:
        xy, kl, trust = fit_cosine(vectors[family], 30, 42)
        path = alternative / f"{family}_cosine_p30_s42.npz"
        np.savez_compressed(path, coordinates=xy, labels=labels, sample_ids=ids, kl_divergence=kl,
                            trustworthiness_original_cosine_k10=trust,
                            features_sha256=digest(out / f"{family}_features.npz"))
        alt["journal_design"]["coordinate_files"][family] = path.name
    save_json(alternative / "report.json", alt)
    review_page(rows, cfg, out)
    b, t = [report["selected"][family]["metrics"] for family in FAMILIES]
    lines_md = ["# VQ16000步最佳FID与较弱普通VQ：700段对比", "",
                "TMR侧精确对应用户提供的2026-10-29 00:51:50、VQ16000步、FID0.0664记录，使用batch512训练的net_best_fid。普通侧仍为net_last。两个模型编码相同700段运动与64帧窗口。", "",
                "|动作|数量|占比|", "|---|---:|---:|"]
    lines_md += [f"|{label}|{n}|{n/700:.2%}|" for label, n in COUNTS.items()]
    lines_md += ["", "最大/最小数量比为112/75≈1.493，单类占比10.71%至16%。跑和舞蹈各112段，其余数量充足类各111段；坐下、踢腿、投掷保留现有可用上限。未用复制、镜像或重叠滑窗补数。", "",
                 "|原始512维宏平均指标|普通net_last|指定TMR-VQ|变化|", "|---|---:|---:|---:|",
                 f"|余弦轮廓系数|{b['macro_silhouette']:.4f}|{t['macro_silhouette']:.4f}|{t['macro_silhouette']-b['macro_silhouette']:+.4f}|",
                 f"|3-NN同类比例|{b['macro_neighbor_purity']:.2%}|{t['macro_neighbor_purity']:.2%}|{100*(t['macro_neighbor_purity']-b['macro_neighbor_purity']):+.2f}个百分点|", "",
                 "数据筛选使用HumanML3D训练与验证，按原始录制去重，27个动作关键词类别参与排他/一致性筛选。类别选择参考432段开发试验，属于事后探索；完整普查、候选组合与原先更严数量约束的方案保留在action_search_700_*目录。最终采用更具体的动作，去掉宽泛举臂类，加入开发试验有正收益的踢腿；数量比约束在700段正式编码前由1.35调整为1.5。", "",
                 f"最终字幕证据分层为{tiers}：pure2为至少两条命中且无竞争动作；dominant2为至少两条且覆盖不少于2/3描述、其他类别最多单条、存在独占描述；pure1为单条命中且无其他类别关键词。所有标签都仍待人工核验。复核页面review.html展示完整骨架运动和对应窗口，不显示模型分数。", "",
                 "主图沿用PCA50→欧氏t-SNE、P30/E12/seed42、最多3000次迭代。完整保留9组配对参数图和原始512维cosine补充图。主指标在PCA之前的512维计算，按类别宏平均，避免大类仅凭数量主导分数。", "",
                 "这是指定较弱普通状态与最佳FID的TMR状态在所选动作集合上的描述性对比，不是双方最佳性能差距、严格单因素消融或独立泛化验证。换动作和改变字幕筛选规则同时改变任务，不能将与旧500段图的分数变化视为模型训练改善。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines_md), encoding="utf-8")
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save_json(out / "verification.json", {
        "700_distinct_original_recordings": True, "class_counts": COUNTS,
        "same_inputs_and_windows_for_both_models": True,
        "requested_vq16k_log_line_and_checkpoint_identity_verified": True,
        "all700_raw_motion_hashes_match": True, "caption_tier_counts": tiers,
        "input_artifacts_unchanged": True, "no_new_training_or_teacher_outputs": True,
        "normalization_sha256": {name: digest(repo_path(cfg["normalization_dir"]) / name) for name in ("mean.npy", "std.npy")},
        "labels_pending_human_review": True,
    })
    print(f"Completed700-clip comparison: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
