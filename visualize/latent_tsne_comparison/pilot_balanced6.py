#!/usr/bin/env python3
"""Compare three declared six-action development cohorts using cached features."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from common import PROJECT, digest, read_rows, save_json, write_rows
from optimize_display import score, select_indices


GROUPS = {
    "A_sit_wave": ["run", "jump", "sit", "kick", "throw", "wave"],
    "B_walk_wave": ["walk", "run", "jump", "kick", "throw", "wave"],
    "C_sit_clap": ["run", "jump", "sit", "kick", "throw", "clap"],
}
SEEDS = (20261029, 20261030, 20261031)
MODELS = ("baseline_best", "tmr_best", "baseline_weaker")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/balanced6_pilot_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Preserve completed diagnostics; use a new output directory")
    parent = PROJECT / "artifacts/checkpoint_contrast_v1"
    audit_path = PROJECT / "artifacts/dense_train_validation_v1/selection_audit.json"
    prior = json.loads((parent / "report.json").read_text())
    audit = json.loads(audit_path.read_text())
    rows = read_rows(parent / "samples.csv")
    labels = np.array([row["label"] for row in rows])
    ids = np.array([row["sample_id"] for row in rows])
    assert len(set(ids)) == len(ids)
    all_sources = np.array([audit["source_path_by_sample_id"][ident] for ident in ids])
    assert len(set(all_sources)) == len(ids)
    inputs = {str(path): digest(path) for path in (
        parent / "report.json", parent / "samples.csv", audit_path,
        *(parent / f"{name}_features.npz" for name in MODELS))}
    plan = {
        "purpose": "Post-hoc exploratory composition pilot; not an independent test",
        "groups": GROUPS, "samples_per_class": 30, "samples_per_cohort": 180,
        "sampling_seeds": list(SEEDS), "primary_models": list(MODELS[:2]),
        "secondary_model": "baseline_weaker",
        "selection": "SHA256(seed:sample_id) within each class; no feature/coordinate access",
        "group_rationale": "A: earlier gain actions plus wave; B: replace scarce sit with walk; C: replace wave with already easy clap",
        "metric": "Cosine silhouette and recording-excluded 3-NN purity in original L2-normalized 512-D post-quantization temporal means",
        "split": "train+val", "labels_human_reviewed": False,
        "class_compositions_informed_by_previous_diagnostics": True,
        "sampling_repeats_are_independent_training_runs": False,
        "sampling_repeats_can_overlap": True,
        "new_inference": False, "new_training": False, "new_test_evaluation": False,
        "new_projection": False, "parent_input_sha256": inputs,
    }
    # Record all compositions and repeat seeds before computing their scores.
    save_json(out / "protocol_before_scoring.json", plan)
    vectors = {}
    for name in MODELS:
        with np.load(parent / f"{name}_features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            assert str(data["checkpoint_sha256"]) == prior["models"][name]["sha256"]
            vectors[name] = data["post_mean"].copy()
        assert vectors[name].shape == (len(rows), 512)
        assert np.isfinite(vectors[name]).all()
        np.testing.assert_allclose(np.linalg.norm(vectors[name], axis=1), 1, atol=1e-6)
    cases, flat = {}, []
    for group, classes in GROUPS.items():
        cases[group] = {}
        for seed in SEEDS:
            idx = select_indices(rows, classes, 30, seed, balanced=True)
            chosen = [rows[i] for i in idx]
            assert Counter(labels[idx]) == Counter({label: 30 for label in classes})
            assert len(set(all_sources[idx])) == 180
            case_path = out / group / str(seed)
            write_rows(case_path / "samples.csv", chosen)
            evaluated = {name: score(vectors[name][idx], labels[idx], all_sources[idx])
                         for name in MODELS}
            primary_delta = {
                key: evaluated["tmr_best"][key] - evaluated["baseline_best"][key]
                for key in ("macro_silhouette", "macro_neighbor_purity")
            }
            case = {"metrics": evaluated, "primary_delta": primary_delta,
                    "manifest_sha256": digest(case_path / "samples.csv"),
                    "class_counts": dict(Counter(labels[idx])),
                    "split_counts": dict(Counter(audit["split_by_sample_id"][ident] for ident in ids[idx])),
                    "original_recordings": 180}
            cases[group][str(seed)] = case
            save_json(case_path / "report.json", case)
            for name, metrics in evaluated.items():
                flat.append({"group": group, "sampling_seed": seed, "model": name,
                             "samples": 180, "macro_silhouette": metrics["macro_silhouette"],
                             "macro_3nn_purity": metrics["macro_neighbor_purity"]})
            print(f"{group}, seed={seed}: baseline {evaluated['baseline_best']['macro_silhouette']:.4f}, "
                  f"TMR {evaluated['tmr_best']['macro_silhouette']:.4f}; "
                  f"3-NN delta {primary_delta['macro_neighbor_purity']*100:+.2f} pp", flush=True)
    summary = {}
    for group, repeats in cases.items():
        summary[group] = {"classes": GROUPS[group], "models": {}, "primary_delta": {}}
        for name in MODELS:
            summary[group]["models"][name] = {}
            for key in ("macro_silhouette", "macro_neighbor_purity"):
                values = [repeat["metrics"][name][key] for repeat in repeats.values()]
                summary[group]["models"][name][key] = {
                    "mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}
        for key in ("macro_silhouette", "macro_neighbor_purity"):
            values = [repeat["primary_delta"][key] for repeat in repeats.values()]
            summary[group]["primary_delta"][key] = {
                "mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}
    save_json(out / "report.json", {"protocol": plan, "summary": summary, "cases": cases})
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    save_json(out / "verification.json", {
        "cases": len(flat) // len(MODELS), "model_scores": len(flat),
        "all_cases_exactly_6_times_30": True, "one_clip_per_recording_in_each_case": True,
        "same_rows_and_windows_across_models": True, "input_cache_ids_labels_and_checkpoint_hashes_match": True,
        "all_feature_vectors_finite_and_unit_norm": True,
        "all_input_files_unchanged": all(digest(Path(path)) == sha for path, sha in inputs.items()),
        "limits": "Reused training-inclusive pool; labels pending; overlapping sampling repeats; no confirmatory inference",
    })


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
