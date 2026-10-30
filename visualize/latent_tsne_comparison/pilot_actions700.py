#!/usr/bin/env python3
"""Explicit post-hoc action-composition pilot; no individual score-based filtering."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from common import PROJECT, REPO, digest, read_rows, save_json, write_rows
from features import FrozenEncoder, motion_batch, normalize
from optimize_display import score


SEED = 20261030
TIERS = {"pure2": 0, "dominant2": 1, "pure1": 2}


def ranked_indices(rows, label, audit, seed=SEED):
    candidates = [i for i, row in enumerate(rows) if row["label"] == label]
    return sorted(candidates, key=lambda i: (
        TIERS[audit["tier_by_sample_id"][rows[i]["sample_id"]]],
        hashlib.sha256(f"{seed}:{rows[i]['sample_id']}".encode()).hexdigest()))


def quotas(classes, available, weighted, target=700):
    weights = {label: 1.08 if label in weighted else 1.0 for label in classes}
    for lower in range(80, target//len(classes)+1):
        if any(available[label] < lower for label in classes):
            continue
        caps = {label: min(available[label], int(np.floor(1.35*lower))) for label in classes}
        if sum(caps.values()) < target:
            continue
        counts = dict.fromkeys(classes, lower)
        for _ in range(target-sum(counts.values())):
            possible = [label for label in classes if counts[label] < caps[label]]
            label = min(possible, key=lambda label: (counts[label]/weights[label], classes.index(label)))
            counts[label] += 1
        assert max(counts.values())/min(counts.values()) <= 1.35
        return counts
    return None


def main():
    parent = PROJECT / "artifacts/action_search_700_merged_v1"
    out = parent / "pilot"
    if out.exists():
        raise FileExistsError("Pilot already exists")
    rows = read_rows(parent / "candidates.csv")
    audit = json.loads((parent / "selection_audit.json").read_text())
    classes = sorted(label for label, n in audit["counts"].items() if n >= 75)
    original = json.loads((PROJECT / "artifacts/selected_checkpoints_500_v1/report.json").read_text())
    states = {family: {k: v for k, v in original["selected"][family].items() if k != "metrics"}
              for family in ("baseline", "tmr_b512")}
    identity_path = REPO / "output/hml3d/GPT_amg_b512_full_bf16/vq_identity.json"
    identity = json.loads(identity_path.read_text())
    assert states["tmr_b512"]["sha256"] == identity["vq_sha256"]
    for state in states.values():
        assert digest(state["path"]) == state["sha256"]
    log_path = REPO / "output/hml3d/GPT_amg_b512_full_bf16/run.log"
    line = [line for line in log_path.read_text().splitlines()
            if "2026-10-29 23:27:02,607" in line and "Eva. Iter 170000" in line and "FID. 0.2698" in line]
    assert len(line) == 1
    plan = {
        "scope": "Post-hoc training-inclusive development pilot; labels pending review",
        "classes": classes, "samples_per_class": 48, "selection_seed": SEED,
        "selection": "caption-evidence tier then SHA256(seed:sample_id), without model-score access",
        "allowed_final_class_counts": [6, 7], "final_target_samples": 700,
        "minimum_final_class_size": 80, "maximum_final_class_ratio": 1.35,
        "mild_weighting": "Top two positive class silhouette gains in this pilot receive1.08 allocation weight; caps may constrain the final ratio",
        "selection_objective": "Largest positive macro silhouette gain among feasible compositions with nonnegative3NN gain and TMR absolute silhouette and purity at least their respective medians over feasible compositions",
        "individual_samples_selected_by_model_score": False,
        "models": states, "requested_gpt_evaluation": line[0], "vq_identity": identity,
        "input_sha256": {str(p): digest(p) for p in (parent / "candidates.csv", parent / "selection_audit.json", identity_path)},
        "feature": "512-D L2-normalized post-quantization temporal mean; identical64-frame crop",
        "independent_confirmation": False, "teacher_features_used": False,
    }
    save_json(out / "protocol_before_encoding.json", plan)
    selected = np.concatenate([ranked_indices(rows, label, audit)[:48] for label in classes])
    pilot_rows = [rows[i] for i in selected]
    write_rows(out / "samples.csv", pilot_rows)
    cfg = json.loads((PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json").read_text())
    raw = motion_batch(pilot_rows, cfg)
    assert raw.shape == (len(pilot_rows), 64, 263) and np.isfinite(raw).all()
    ids = np.array([row["sample_id"] for row in pilot_rows])
    labels = np.array([row["label"] for row in pilot_rows])
    sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    assert len(set(sources)) == len(ids)
    features, overall = {}, {}
    torch.set_num_threads(1)
    for family, state in states.items():
        model = FrozenEncoder(cfg, state["path"]).cpu().eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(raw), 16):
                chunks.append(model(torch.from_numpy(raw[start:start+16]))[1].numpy())
        features[family] = normalize(np.concatenate(chunks))
        np.savez_compressed(out / f"{family}_features.npz", post_mean=features[family], labels=labels,
                            sample_ids=ids, checkpoint_sha256=state["sha256"])
        overall[family] = score(features[family], labels, sources)
        del model
        print(f"Pilot encoded {family}: {len(ids)} clips, {len(classes)} actions", flush=True)
    class_gains = {label: overall["tmr_b512"]["per_class"][label]["silhouette"]-
                   overall["baseline"]["per_class"][label]["silhouette"] for label in classes}
    cases = []
    for k in plan["allowed_final_class_counts"]:
        for combination in itertools.combinations(classes, k):
            chosen = list(combination)
            weighted = [label for label in sorted(chosen, key=lambda label: -class_gains[label])[:2]
                        if class_gains[label] > 0]
            allocation = quotas(chosen, audit["counts"], weighted)
            if allocation is None:
                continue
            idx = np.flatnonzero(np.isin(labels, chosen))
            metrics = {family: score(x[idx], labels[idx], sources[idx]) for family, x in features.items()}
            b, t = metrics["baseline"], metrics["tmr_b512"]
            cases.append({"classes": chosen, "allocation": allocation, "slightly_upweighted_classes": weighted,
                          "metrics": metrics, "delta_silhouette": t["macro_silhouette"]-b["macro_silhouette"],
                          "delta_3nn": t["macro_neighbor_purity"]-b["macro_neighbor_purity"]})
    assert cases
    med_s = float(np.median([c["metrics"]["tmr_b512"]["macro_silhouette"] for c in cases]))
    med_p = float(np.median([c["metrics"]["tmr_b512"]["macro_neighbor_purity"] for c in cases]))
    eligible = [c for c in cases if c["delta_silhouette"] > 0 and c["delta_3nn"] >= 0
                and c["metrics"]["tmr_b512"]["macro_silhouette"] >= med_s
                and c["metrics"]["tmr_b512"]["macro_neighbor_purity"] >= med_p]
    if not eligible:
        save_json(out / "report.json", {"plan": plan, "overall": overall, "class_gains": class_gains,
                  "all_feasible_compositions": cases, "selected": None, "reason": "No composition meets the declared criteria"})
        raise RuntimeError("Do not force an unsupported favorable selection")
    chosen = max(eligible, key=lambda c: (c["delta_silhouette"], c["delta_3nn"], c["metrics"]["tmr_b512"]["macro_silhouette"]))
    save_json(out / "report.json", {"plan": plan, "overall": overall, "class_gains": class_gains,
              "all_feasible_compositions": cases, "tmr_absolute_medians": {"silhouette": med_s, "purity": med_p},
              "selected": chosen, "selection_uses_2d_coordinates": False})
    print(json.dumps({"class_gains": class_gains, "feasible_compositions": len(cases), "chosen_classes": chosen["classes"],
                      "allocation": chosen["allocation"], "pilot_delta_silhouette": chosen["delta_silhouette"],
                      "pilot_delta_3nn_pp": chosen["delta_3nn"]*100}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
