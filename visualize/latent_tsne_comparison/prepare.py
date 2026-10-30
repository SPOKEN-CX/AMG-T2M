"""Caption-only screening; no model features are used in sample selection."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from common import digest, read_rows, repo_path, save_json, source_id, validate_rows, write_rows


def captions_for(path, length, fps):
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit("#", 3)
        if len(parts) != 4:
            raise ValueError(f"Malformed caption line in {path}")
        text, _, begin, end = parts
        begin, end = float(begin), float(end)
        missing_timestamps = math.isnan(begin) or math.isnan(end)
        # Match dataset_TM_eval.py: HumanML3D NaN timestamp fields mean 0.
        begin = 0.0 if math.isnan(begin) else begin
        end = 0.0 if math.isnan(end) else end
        if not (math.isfinite(begin) and math.isfinite(end)):
            raise ValueError(f"Nonfinite caption timestamps: {path}")
        if begin == end == 0:
            lo, hi = 0, length
        else:
            if begin < 0 or end <= begin:
                raise ValueError(f"Invalid caption timestamps: {path}")
            lo, hi = max(0, int(begin * fps)), min(length, int(end * fps))
        if hi > lo:
            result.append({"text": text.strip(), "start": lo, "end": hi,
                           "nan_timestamp_coerced_to_zero": missing_timestamps})
    return result


def rank_key(row, cfg):
    # Quality rules precede an outcome-independent, stable seeded shuffle.
    random_key = hashlib.sha256(f"{cfg['selection_seed']}:{row['motion_id']}".encode()).hexdigest()
    risk_count = len([s for s in row["risk_flags"].split(";") if s])
    return (risk_count, -min(2, int(row["caption_votes"])), random_key)


def prepare(cfg, out, force=False):
    out = Path(out)
    manifest = out / "samples.csv"
    if manifest.exists() and not force:
        raise FileExistsError(f"Preserving existing selection: {manifest}. Use a new output directory or --force.")
    root = repo_path(cfg["dataset_root"])
    patterns = {c["name"]: re.compile(c["pattern"], re.I) for c in cfg["classes"]}
    risks = re.compile(cfg["screening_risk_pattern"], re.I)
    ids = (root / "test.txt").read_text().split()
    # Prefer non-mirrored IDs. The canonical ID can only occur once in the pool.
    canonical = {}
    for ident in sorted(ids, key=lambda s: (s.startswith("M"), s)):
        canonical.setdefault(source_id(ident), ident)
    rows, excluded = [], Counter()
    timestamp_coercions = 0
    invalid_motions = []
    width = cfg["window_size"]
    for original, ident in canonical.items():
        path = root / "new_joint_vecs" / f"{ident}.npy"
        cap_path = root / "texts" / f"{ident}.txt"
        if not path.is_file() or not cap_path.is_file():
            raise FileNotFoundError(f"Missing test motion/text: {ident}")
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if x.ndim != 2 or x.shape[1] != cfg["architecture"]["input_dim"] or not np.isfinite(x).all():
            excluded["invalid_motion_array"] += 1
            invalid_motions.append({"motion_id": ident, "shape": list(x.shape),
                                    "nonfinite_values": int((~np.isfinite(x)).sum())})
            continue
        length = len(x)
        if length < width:
            excluded["shorter_than_window"] += 1
            continue
        caps = captions_for(cap_path, length, cfg["fps"])
        timestamp_coercions += sum(c["nan_timestamp_coerced_to_zero"] for c in caps)
        matches = {name: [c for c in caps if pat.search(c["text"])] for name, pat in patterns.items()}
        labels = [name for name, group in matches.items() if group]
        if len(labels) != 1:
            excluded["no_target_label" if not labels else "multiple_target_labels"] += 1
            continue
        label = labels[0]
        usable = [c for c in matches[label] if c["end"] - c["start"] >= width]
        if not usable:
            excluded["matching_caption_segment_too_short"] += 1
            continue
        # Prefer a tagged segment if available, then a short matching description.
        chosen = min(usable, key=lambda c: (c["end"] - c["start"], len(c["text"]), c["text"]))
        lo, hi = chosen["start"], chosen["end"]
        start = lo + (hi - lo - width) // 2
        end = start + width
        flags = []
        overlapping = [c for c in caps if c["start"] < end and c["end"] > start]
        if any(risks.search(c["text"]) for c in overlapping):
            flags.append("other_action_keyword")
        if any(re.search(r"\b(?:then|afterwards|subsequently)\b", c["text"], re.I) for c in overlapping):
            flags.append("possible_action_sequence")
        if len(matches[label]) < 2:
            flags.append("single_caption_vote")
        rows.append({"sample_id": f"{ident}_{start:04d}_{end:04d}", "motion_id": ident,
                     "source_id": original, "label": label, "start_frame": start, "end_frame": end,
                     "motion_frames": length, "annotation_start": lo, "annotation_end": hi,
                     "caption": chosen["text"], "captions_json": json.dumps(caps, ensure_ascii=False),
                     "caption_votes": len(matches[label]), "risk_flags": ";".join(flags),
                     "review_status": "pending", "review_note": "", "motion_sha256": digest(path)})
    groups = {c["name"]: sorted([r for r in rows if r["label"] == c["name"]], key=lambda r: rank_key(r, cfg))
              for c in cfg["classes"]}
    count = min(cfg["samples_per_class"], *(len(g) for g in groups.values()))
    if count < 2:
        raise ValueError(f"Insufficient candidates: { {k: len(v) for k, v in groups.items()} }")
    selected = [r for group in groups.values() for r in group[:count]]
    counts = validate_rows(selected, cfg)
    write_rows(out / "candidates.csv", [r for group in groups.values() for r in group])
    write_rows(manifest, selected)
    report = {"split": "test", "test_ids": len(ids), "canonical_motion_ids": len(canonical),
              "candidate_counts": {k: len(v) for k, v in groups.items()}, "selected_counts": counts,
              "excluded_counts": dict(excluded), "selection_seed": cfg["selection_seed"],
              "nan_timestamp_fields_coerced_to_zero_captions": timestamp_coercions,
              "invalid_motions_excluded": invalid_motions,
              "selection_rule": "single target class across captions; valid 64-frame caption segment; risk flags ascending; caption votes (capped at 2) descending; seeded SHA256 shuffle; center crop",
              "labels": "caption-screened; all pending human review",
              "source_grouping": "canonical HumanML3D ID (M prefix removed); original AMASS source index is unavailable",
              "selected_risk_flags": dict(Counter(flag for r in selected for flag in r["risk_flags"].split(";") if flag)),
              "manifest_sha256": digest(manifest), "test_split_sha256": digest(root / "test.txt")}
    save_json(out / "selection_audit.json", report)
    save_json(out / "config_snapshot.json", cfg)
    print(f"Prepared {len(selected)} samples ({count}/class), labels pending review", flush=True)
    return manifest


def finalize(cfg, reviewed_path, out):
    rows = read_rows(reviewed_path)
    accepted = [r for r in rows if r["review_status"] == "accepted"]
    groups = {c["name"]: sorted([r for r in accepted if r["label"] == c["name"]], key=lambda r: rank_key(r, cfg))
              for c in cfg["classes"]}
    n = min(cfg["samples_per_class"], *(len(g) for g in groups.values()))
    if n < 2:
        raise ValueError(f"Need >=2 accepted samples per class: { {k: len(v) for k, v in groups.items()} }")
    selected = [r for group in groups.values() for r in group[:n]]
    counts = validate_rows(selected, cfg, require_reviewed=True)
    out = Path(out)
    if (out / "samples.csv").exists():
        raise FileExistsError("Use a new output directory for each reviewed selection")
    write_rows(out / "samples.csv", selected)
    save_json(out / "selection_audit.json", {"reviewed_input_sha256": digest(reviewed_path),
              "accepted_counts": {k: len(v) for k, v in groups.items()}, "selected_counts": counts,
              "manifest_sha256": digest(out / "samples.csv"), "labels": "human-accepted via supplied review manifest"})
    save_json(out / "config_snapshot.json", cfg)
    print(f"Finalized {len(selected)} accepted samples ({n}/class)", flush=True)
    return out / "samples.csv"
