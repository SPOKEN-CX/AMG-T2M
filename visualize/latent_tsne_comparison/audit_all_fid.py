#!/usr/bin/env python3
"""Inventory local FID records and compare the lowest historical contrastive VQ on fixed clips.

Historical test-selected models are kept separate from validation-selected AMG models.
No reconstruction FID is calculated on the action visualization subset.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, read_rows, repo_path, save_json
from compare_gpt_linked_vq import FAMILIES, SETTINGS, cohort
from features import FrozenEncoder, motion_batch, normalize


LEGACY = "checkpoints/self_train/VQVAE/t2m/run.log"


def parse_log(path):
    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    relative = path.relative_to(REPO).as_posix()
    configurations = []
    for match in re.finditer(r"INFO (\{)", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start(1):])
        except ValueError:
            continue
        configurations.append(value)
    events = []
    current = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r"Eva\. Iter (\d+)\s*:, FID\. ([\d.eE+-]+)", line)
        if match:
            current = {"iteration": int(match[1]), "fid": float(match[2]),
                       "line": number, "text": line}
            events.append(current)
        improved = re.search(r"FID Improved from \S+ to (\S+)", line)
        if improved and current is not None:
            current["fid_improvement_precision"] = float(improved[1])
    cfg = configurations[0] if configurations else {}
    if "amg_audit" in relative:
        category = "fixed_validation_reevaluation"
    elif "_eval/" in relative or relative.startswith("results/"):
        category = "test_reevaluation_not_selection"
    elif relative == LEGACY:
        category = "legacy_contrastive_VQ_test_selected"
    elif path.name == "initial_10000_run.log":
        category = "AMG_VQ_validation_initial_segment"
    elif path.parent.name.startswith("VQ_amg_"):
        category = "AMG_VQ_validation" if cfg.get("total_iter", 0) >= 1000 else "AMG_VQ_smoke"
    elif cfg.get("dataname") == "BABEL_TEACH":
        category = "BABEL_different_dataset"
    elif "GPT" in relative or "VQTransformer" in relative:
        category = "GPT_generation_not_VQ_reconstruction"
    elif "Length_" in relative:
        category = "length_estimator_no_VQ_FID"
    else:
        category = "other_VQ_or_unidentified"
    best = min(events, key=lambda row: row["fid"]) if events else None
    return {"log": relative, "sha256_of_read_snapshot": hashlib.sha256(payload).hexdigest(),
            "category": category, "configurations": configurations, "evaluation_count": len(events),
            "minimum_logged_event": best, "events": events,
            "contains_contrastive_training_loss": "Contrastive:" in text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/all_fid_audit_700_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Preserve previous results; choose a new output directory")
    listing = subprocess.run([
        "rg", "--files", "--hidden", "--no-ignore", "-g", "run.log", "-g", "initial_10000_run.log",
        "-g", "!dataset/**", "-g", "!.git/**", "-g", "!venv/**", "-g", "!envs/**",
    ], cwd=REPO, capture_output=True, text=True, check=True)
    logs = [parse_log(REPO / name) for name in sorted(listing.stdout.splitlines())]
    comparable = []
    for run in logs:
        if run["category"] != "AMG_VQ_validation":
            continue
        path = REPO / run["log"]
        summary_path = path.parent / "result_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            fid, iteration = summary["best_fid"], summary["best_iteration"]
        else:
            manifest = json.loads((REPO / "output/hml3d/VQ_amg_frozen/manifest.json").read_text())
            assert path.parent.name == "VQ_amg_full"
            fid, iteration = manifest["best_validation_fid"], manifest["best_iteration"]
        assert any(event["iteration"] == iteration and abs(event["fid"]-fid) < .000051 for event in run["events"])
        checkpoint = path.parent / "net_best_fid.pth"
        comparable.append({"run": path.parent.name, "batch_size": run["configurations"][0]["batch_size"],
                           "minimum_validation_fid": fid, "iteration": iteration,
                           "checkpoint": str(checkpoint), "sha256": digest(checkpoint)})
    comparable.sort(key=lambda row: row["minimum_validation_fid"])
    winner = comparable[0]
    assert winner["iteration"] == 16000 and winner["batch_size"] == 512
    legacy = next(run for run in logs if run["log"] == LEGACY)
    assert legacy["contains_contrastive_training_loss"]
    historical = min([run for run in logs if run["category"] in (
        "AMG_VQ_validation", "AMG_VQ_smoke", "legacy_contrastive_VQ_test_selected")],
        key=lambda row: row["minimum_logged_event"]["fid"])
    assert historical["log"] == LEGACY
    legacy_checkpoint = REPO / LEGACY.replace("run.log", "net_best_fid.pth")
    source = PROJECT / "artifacts/bestfid16k_vs_weaker_700_v1"
    prior = json.loads((source / "report.json").read_text())
    plan = json.loads((source / "protocol_before_encoding.json").read_text())
    base = copy.deepcopy(prior["selected"]["baseline"])
    base.pop("metrics")
    assert digest(base["path"]) == base["sha256"]
    assert winner["sha256"] == prior["selected"]["tmr_b512"]["sha256"]
    pinned_inputs = {str(source / name): digest(source / name) for name in (
        "samples.csv", "report.json", "selection_audit.json", "baseline_features.npz", "final_checks.json")}
    fixed_path = REPO / "output/amg_audit/20261029/fixed_validation_b1024/comparison.json"
    fixed = json.loads(fixed_path.read_text())
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "repository": str(REPO),
        "search": "All run.log and initial_10000_run.log files; rg includes ignored and hidden paths",
        "search_excludes": ["dataset/**", ".git/**", "venv/**", "envs/**"],
        "log_count": len(logs), "evaluation_count_in_read_snapshots": sum(run["evaluation_count"] for run in logs),
        "logs": logs, "comparable_AMG_validation_ranking": comparable,
        "validation_selection": winner,
        "fixed_validation_reevaluation": {"source": str(fixed_path), "report": fixed,
            "note": "Same b512 16k model; 0.062524 is a fixed-input re-evaluation, not different parameters"},
        "lowest_historical_contrastive_VQ_log": {k: v for k, v in historical.items() if k != "events"},
        "legacy_checkpoint": {"path": str(legacy_checkpoint), "sha256": digest(legacy_checkpoint),
            "directory_log_minimum": legacy["minimum_logged_event"],
            "iteration_metadata_in_weights": False,
            "identity_limit": "Weights contain only net. Directory association is known; exact184000-step identity cannot be independently proved without an archived numbered checkpoint."},
        "legacy_protocol_evidence": {
            "train_vq_constra.py": "DATALoader(args.dataname, True,32,...)",
            "src/datasets/dataset_TM_eval.py": "is_test=True selects test.txt; False selects val.txt",
            "note": "Available legacy source indicates test-based selection; historical source snapshot and run invocation are not preserved. Do not call0.0301 a comparable validation FID."},
        "baseline": base, "samples": 700, "class_counts": prior["class_counts"],
        "original_validation_comparison": str(source), "pinned_input_sha256": pinned_inputs,
        "selection_scope": "User-requested weaker ordinary net_last versus lowest historical-FID contrastive VQ; validation-selected reference retained separately",
    }
    save_json(out / "audit.json", audit)
    with (out / "fid_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["log", "category", "evaluations", "minimum_logged_fid", "iteration", "line"])
        for run in logs:
            event = run["minimum_logged_event"] or {}
            writer.writerow([run["log"], run["category"], run["evaluation_count"],
                             *[event.get(key, "") for key in ("fid", "iteration", "line")]])
    print(json.dumps({"comparable_validation_ranking": comparable,
                      "historical_minimum": legacy["minimum_logged_event"]}, ensure_ascii=False), flush=True)

    # A separate illustration meets the all-directory request without mislabeling test FID.
    rows = read_rows(source / "samples.csv")
    ids = np.array([row["sample_id"] for row in rows])
    labels = np.array([row["label"] for row in rows])
    selection = json.loads((source / "selection_audit.json").read_text())
    sources = np.array([selection["source_path_by_sample_id"][ident] for ident in ids])
    cfg = json.loads((PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json").read_text())
    raw = motion_batch(rows, cfg)
    torch.set_num_threads(1)
    with np.load(source / "baseline_features.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sample_ids"], ids)
        np.testing.assert_array_equal(data["labels"], labels)
        assert str(data["checkpoint_sha256"]) == base["sha256"]
        baseline = data["post_mean"].copy()
    legacy_state = {"name": "legacy_net_best_fid", "path": str(legacy_checkpoint),
        "sha256": audit["legacy_checkpoint"]["sha256"], "vq_iteration": None,
        "directory_log_best_iteration": legacy["minimum_logged_event"]["iteration"],
        "directory_log_best_fid": legacy["minimum_logged_event"]["fid"],
        "fid_protocol": "legacy test-selected historical record; not comparable validation FID",
        "exact_numbered_checkpoint_identity_verified": False}
    model = FrozenEncoder(cfg, legacy_checkpoint).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            chunks.append(model(torch.from_numpy(raw[start:start+16]))[1].numpy())
            if start % 160 == 0:
                print(f"Legacy checkpoint encoding: {min(start+16, len(rows))}/700", flush=True)
    del model
    vectors = {"baseline": baseline, "tmr_b512": normalize(np.concatenate(chunks))}
    study = out / "legacy_lowest_record_vs_netlast_700"
    report = cohort(study, rows, sources, vectors,
                    {"baseline": base, "tmr_b512": legacy_state}, SETTINGS, classes=list(prior["class_counts"]))
    for key in ("macro_silhouette", "macro_neighbor_purity"):
        np.testing.assert_allclose(report["selected"]["baseline"]["metrics"][key],
                                   prior["selected"]["baseline"]["metrics"][key], atol=1e-12)
    report["selection_protocol"] = {"original700_protocol": plan,
        "checkpoint_selection_audit": str(out / "audit.json"),
        "legacy_training_source_snapshot_missing": True,
        "historical_fids_not_compared_across_protocols": True}
    report["journal_design"].update({
        "context": "Historical checkpoint contrast  |  700 paired clips  |  7 actions",
        "titles": ["(a)  Baseline VQ-VAE", "(b)  Legacy contrastive VQ-VAE"],
        "subtitles": {"baseline": "net_last | VQ step 300,000 | validation FID 0.1229",
                      "tmr_b512": "net_best_fid | legacy log minimum FID 0.0301*"},
        "marker_size": 4.2, "legend_columns": 4,
        "footer_left": "*Legacy test-based record; exact step identity unverified.",
        "footer_bottom": "Original 512-D macro metrics. Post-hoc train + val action subset; caption labels pending review.",
        "scope": "Historical lowest recorded FID, mixed protocols; separate from validation-selected AMG comparison",
        "scope_note": "\n旧版同目录日志最低FID0.0301；现存权重仅含net，无法独立验证184000步身份。可用旧源码使用test.txt评估；因此本图为历史权重对比，不把0.0301标作与当前模型可比的验证FID。\n"})
    save_json(study / "report.json", report)
    b, t = [report["selected"][family]["metrics"] for family in FAMILIES]
    text = ["# 全目录FID核查与700段权重对比", "",
        f"扫描{len(logs)}份主训练/评估日志，解析{audit['evaluation_count_in_read_snapshots']}条FID记录。run.log包含完整续训日志，initial_10000_run.log补齐batch512前10000步。GPT生成、BABEL、测试复测及冒烟试验分别标识，详见fid_inventory.csv与audit.json。", "",
        "## 同口径AMG-VQ训练验证排名", "", "|batch|最佳步数|验证FID|", "|---:|---:|---:|"]
    text += [f"|{r['batch_size']}|{r['iteration']}|{r['minimum_validation_fid']:.8f}|" for r in comparable]
    text += ["", "同口径验证最低仍是batch512、16000步，FID0.06643728。固定1504段验证输入复测为0.06252396，是同一组权重；batch1024固定输入为0.07256454，batch128为0.12338640。固定验证对照中的普通模型使用net_best_fid，并非本次指定的net_last，不将该行FID移用于net_last。", "",
        "原有700段图恰好已经使用这个验证赢家与指定net_last，权重SHA256一致，因此无需重新推理或重选t-SNE参数。原图及完整检查保留在：", str(source / "journal_style_v1/journal_comparison.pdf"), "",
        "## 全目录更低历史记录：单独呈现", "",
        "checkpoints/self_train/VQVAE/t2m/run.log包含对比学习训练，最低记录为2025-01-13、184000步、FID0.0301，同目录有net_best_fid.pth。可用旧版train_vq_constra.py使用DATALoader(..., True,...)，在当前数据加载器中对应test.txt；原始运行命令和源码快照缺失。因此它属于旧版、疑似测试集选权重的记录，不能取代当前验证FID排名。", "",
        "为覆盖用户提出的全目录范围，额外用这份旧版权重与同一普通net_last编码完全相同的700段、相同64帧窗口，生成独立图。图中明确写Legacy contrastive，并注明历史FID口径及步数身份限制；未把旧版包装成新训练的TMR-VQ。权重只含net而不含迭代元数据，也没有184000步归档文件，故目录日志与权重之间的精确步数关系尚不能独立证明。", "",
        "|原始512维宏平均指标|普通net_last|旧版最低记录目录权重|", "|---|---:|---:|",
        f"|余弦轮廓系数|{b['macro_silhouette']:.4f}|{t['macro_silhouette']:.4f}|",
        f"|3-NN同类比例|{b['macro_neighbor_purity']:.2%}|{t['macro_neighbor_purity']:.2%}|", "",
        "FID衡量重建分布，不能保证动作潜空间可分性同步更好。两组图均保留全部700段、7类和9组配对投影参数；不按模型分数更换样本、不删点或移动类别。数据来自train+val、字幕标签待人工审核、动作集合来自此前开发筛选。这是所选权重与动作集合上的探索对比，不能作为双方最佳模型的严格消融结论。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(text), encoding="utf-8")
    assert pinned_inputs == {path: digest(path) for path in pinned_inputs}
    assert digest(legacy_checkpoint) == legacy_state["sha256"]
    assert digest(base["path"]) == base["sha256"]
    save_json(out / "verification.json", {"status": "passed", "samples": len(rows),
        "same700_manifest": digest(study / "samples.csv") == digest(source / "samples.csv"),
        "same_baseline_features_and_original_space_scores": True,
        "same_projection_settings": SETTINGS, "all_prior_inputs_unchanged": True,
        "validation_winner_is_unchanged": True, "legacy_protocol_and_identity_limit_disclosed": True,
        "new_training": False, "teacher_outputs_used": False})
    print(f"Finished audit and historical700 comparison: {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
