#!/usr/bin/env python3
"""Equal nominal checkpoint budgets, fixed primary features, validation data only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, repo_path, save_json, read_rows, validate_rows
from features import FrozenEncoder, motion_batch, normalize
from expand_actions import ANNOTATIONS, macro_metrics, make_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT / "artifacts/expanded_validation_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    study = args.study.resolve()
    out = study / "checkpoint_probe"
    if (out / "report.json").exists():
        raise FileExistsError("Checkpoint probe already completed")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((study / "config_snapshot.json").read_text())
    rows = read_rows(study / "samples.csv")
    validate_rows(rows, cfg, require_balance=False)
    if args.resume and (out / "protocol_before_encoding.json").exists():
        prior = json.loads((out / "protocol_before_encoding.json").read_text())
        if prior["cohort_sha256"] != digest(study / "samples.csv"):
            raise ValueError("Cannot resume after changing the validation cohort")
    study_report = json.loads((study / "report.json").read_text())
    if study_report["manifest_sha256"] != digest(study / "samples.csv"):
        raise ValueError("Cached validation features have a stale cohort")
    candidates = {
        "baseline": [{"name": n.removesuffix(".pth"), "path": str(REPO / "output/hml3d/VQ_base" / n)}
                     for n in ["net_best_fid.pth", "net_best_matching.pth", "net_best_top1.pth", "net_best_div.pth", "net_last.pth"]],
        "tmr_b512": [{"name": "net_best_fid", "path": str(REPO / "output/hml3d/VQ_amg_b512_k50_100k/net_best_fid.pth")}]
                     + [{"name": f"model_{step}", "path": str(REPO / f"checkpoints_model/VQ_amg_b512_k50_100k/model_{step}.pth")}
                        for step in [30000, 50000, 75000, 100000]]}
    for values in candidates.values():
        for item in values:
            if not Path(item["path"]).is_file():
                raise FileNotFoundError(item["path"])
            item["sha256"] = digest(item["path"])
    save_json(out / "protocol_before_encoding.json", {"cohort_sha256": digest(study / "samples.csv"),
        "split": "val", "feature": "post_mean", "selection": "maximize macro silhouette, tie-break macro 3-NN purity independently for each model family",
        "candidates_per_family": 5, "candidates": candidates,
        "limitation": "Available saved checkpoints have different training trajectories and budgets; equal candidate count does not establish a causal ablation"})
    annotations = json.loads(ANNOTATIONS.read_text())
    x = motion_batch(rows, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    all_results, features = {}, {}
    previous = json.loads((out / "partial_results.json").read_text()) if args.resume and (out / "partial_results.json").exists() else {}
    for family, values in candidates.items():
        all_results[family] = []
        features[family] = {}
        for item in values:
            cached = next((r for r in previous.get(family, []) if r["sha256"] == item["sha256"]), None)
            vector_path = out / f"{family}_{item['name']}_features.npz"
            if cached is not None:
                all_results[family].append({**item, "metrics": cached["metrics"]})
                if vector_path.exists():
                    with np.load(vector_path, allow_pickle=False) as data:
                        features[family][item["name"]] = data["features"].copy()
                elif item["name"] == "net_best_fid":
                    with np.load(study / f"{family}_features.npz", allow_pickle=False) as data:
                        features[family][item["name"]] = data["post_mean"].copy()
                print(f"Reused validation metric: {family}/{item['name']}", flush=True)
                continue
            # Reuse the primary representation from the original validation study.
            if item["name"] == "net_best_fid" and item["sha256"] == study_report["checkpoints"][family]:
                with np.load(study / f"{family}_features.npz", allow_pickle=False) as stored:
                    if not np.array_equal(stored["sample_ids"], [r["sample_id"] for r in rows]):
                        raise ValueError("Validation sample order changed")
                    vector = stored["post_mean"].copy()
            else:
                model = FrozenEncoder(cfg, item["path"]).to(device).eval()
                chunks = []
                with torch.inference_mode():
                    for offset in range(0, len(x), 16):
                        chunks.append(model(torch.from_numpy(x[offset:offset + 16]).to(device))[1].cpu().numpy())
                vector = normalize(np.concatenate(chunks))
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            metric = macro_metrics(vector, rows, annotations, 3)
            np.savez_compressed(vector_path, features=vector, sample_ids=np.array([r["sample_id"] for r in rows]), checkpoint_sha256=np.array(item["sha256"]))
            all_results[family].append({**item, "metrics": metric})
            features[family][item["name"]] = vector
            save_json(out / "partial_results.json", all_results)
            print(f"{family}/{item['name']}: silhouette={metric['macro_silhouette']:.4f}, 3-NN={metric['macro_neighbor_purity']:.1%}", flush=True)
    selected = {family: max(values, key=lambda r: (r["metrics"]["macro_silhouette"], r["metrics"]["macro_neighbor_purity"]))
                for family, values in all_results.items()}
    for family, result in selected.items():
        if result["name"] not in features[family]:
            model = FrozenEncoder(cfg, result["path"]).to(device).eval()
            chunks = []
            with torch.inference_mode():
                for offset in range(0, len(x), 16):
                    chunks.append(model(torch.from_numpy(x[offset:offset+16]).to(device))[1].cpu().numpy())
            features[family][result["name"]] = normalize(np.concatenate(chunks))
            del model
    a, b = selected["baseline"]["metrics"], selected["tmr_b512"]["metrics"]
    report = {"split": "val", "independent_test_performed": False, "feature": "post_mean", "samples": len(rows),
              "all_candidates": all_results, "selected": selected,
              "unique_weight_counts": {family: len({item["sha256"] for item in values}) for family, values in candidates.items()},
              "delta_silhouette": b["macro_silhouette"] - a["macro_silhouette"],
              "delta_purity": b["macro_neighbor_purity"] - a["macro_neighbor_purity"],
              "cohort_sha256": digest(study / "samples.csv")}
    save_json(out / "report.json", report)
    for family, result in selected.items():
        np.savez_compressed(out / f"{family}_selected_features.npz", post_mean=features[family][result["name"]],
                            sample_ids=np.array([r["sample_id"] for r in rows]), checkpoint_sha256=np.array(result["sha256"]))
    selected_cfg = copy.deepcopy(cfg)
    for spec in selected_cfg["models"]:
        result = selected[spec["name"]]
        spec["checkpoint"] = result["path"]
        spec["title"] += "\n" + result["name"]
    make_plot(selected_cfg, rows,
              {family: {"post_mean": features[family][result["name"]]} for family, result in selected.items()},
              {"post_mean": {"baseline": a, "tmr": b}}, "post_mean", out)
    lines = ["# 已有checkpoint的语义诊断", "", "固定10类验证样本及量化后时间均值表示，两边各检查5份已保存权重。不是新训练，也没有重新查看测试分数。", "",
             "|模型|权重|宏平均Silhouette|宏平均3-NN|", "|---|---|---:|---:|"]
    for family, values in all_results.items():
        lines += [f"|{family}|{r['name']}|{r['metrics']['macro_silhouette']:.4f}|{r['metrics']['macro_neighbor_purity']:.1%}|" for r in values]
    lines += ["", f"验证选择：普通={selected['baseline']['name']}；TMR={selected['tmr_b512']['name']}。",
              f"差值：Silhouette {report['delta_silhouette']:+.4f}；3-NN {100*report['delta_purity']:+.2f} 个百分点。", "",
              "两侧独立选择同一验证指标；候选文件数相同，去重后的真实权重数量见report.json。训练迭代、保存时机和训练配置不同，不能把这组诊断当作严格TMR因果消融。", "",
              "验证选模后的图仅作探索展示，标签待核验，仍需新数据上的独立确认。", ""]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
