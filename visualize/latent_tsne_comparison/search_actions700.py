#!/usr/bin/env python3
"""Caption-first census of additional action labels for a near-balanced 700-clip study."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import PROJECT, digest, save_json, write_rows
from expand_actions import ANNOTATIONS, collect, recording


EXTRA = [
    {"name": "dance", "pattern": r"\bdanc(?:e|es|ing|ed)\b"},
    {"name": "bend", "pattern": r"\b(?:bend|bends|bending|bent)\b"},
    {"name": "reach", "pattern": r"\breach(?:es|ing|ed)?\b"},
    {"name": "stretch", "pattern": r"\bstretch(?:es|ing|ed)?\b"},
    {"name": "balance", "pattern": r"\bbalanc(?:e|es|ing|ed)\b|\bstand(?:s|ing)?\s+on\s+(?:one|a single)\s+(?:leg|foot)\b"},
    {"name": "lift_object", "pattern": r"\b(?:lift(?:s|ing|ed)?|pick(?:s|ing|ed)?\s+up)\s+(?:(?:up|an?|the)\s+)*(?:box|object|weight|something|item|bag)\b"},
    {"name": "push", "pattern": r"\bpush(?:es|ing|ed)?\b"},
    {"name": "pull", "pattern": r"\bpull(?:s|ing|ed)?\b"},
    {"name": "lie", "pattern": r"\b(?:lie|lies|lying)\b|\blay(?:s|ing)?\s+down\b"},
]

BODY_LABELS = [
    {"name": "arm_raise", "pattern": r"\b(?:rais(?:e|es|ing)|lift(?:s|ing)?)\b[^.!?]{0,30}\b(?:arms?|hands?)\b|\b(?:arms?|hands?)\b[^.!?]{0,24}\b(?:rais(?:e|es|ing)|lift(?:s|ing)?)\b"},
    {"name": "arm_circle", "pattern": r"\b(?:circl\w*|circular|rotat\w*|windmill\w*)\b[^.!?]{0,40}\b(?:arms?|shoulders?)\b|\b(?:arms?|shoulders?)\b[^.!?]{0,35}\b(?:circl\w*|circular|rotat\w*|windmill\w*)\b"},
    {"name": "stand_up", "pattern": r"\b(?:stand(?:s|ing)?|stood|get(?:s|ting)?|got)\s+up\b|\bris(?:e|es|ing)\s+to\s+(?:their|his|her)\s+feet\b"},
    {"name": "stairs", "pattern": r"\b(?:stairs?|staircase|upstairs|downstairs)\b"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/action_search_700_v1")
    parser.add_argument("--minimum-votes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--body-labels", action="store_true")
    parser.add_argument("--dominant", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory")
    cfg = json.loads((PROJECT / "artifacts/dense_train_validation_v1/config_snapshot.json").read_text())
    catalog = json.loads((PROJECT / "expanded_protocol.json").read_text())["classes"] + EXTRA
    if args.body_labels:
        catalog += BODY_LABELS
        for c in catalog:
            if c["name"] == "bend":
                c["pattern"] = r"\bbend(?:s|ing)?\s+(?:forward|forwards|over|down)\b|\bbent\s+(?:forward|over)\b|\bbend(?:s|ing)?\s+(?:at\s+)?(?:the|their|his|her)\s+waist\b"
    protocol = {"classes": catalog, "window_size": 64, "seed": 20261030, "minimum_matching_captions": args.minimum_votes}
    protocol["dominant_caption_rule"] = args.dominant
    plan = {
        "phase": "Availability census, before computing new model scores",
        "catalog": catalog, "splits": ["train", "val"], "test_outcomes_used": False,
        "minimum_matching_captions": args.minimum_votes, "exclude_action_sequence_flags": True,
        "dominant_caption_rule": args.dominant,
        "one_clip_per_original_recording": True, "window_size": 64,
        "sample_selection_uses_model_features": False,
        "target_followup": {"total": 700, "desired_classes": [6, 7, 8],
                            "preferred_max_to_min_count_ratio": 1.25, "maximum_count_ratio": 1.35},
        "annotations_sha256": digest(ANNOTATIONS),
        "note": "Action categories may be selected later using explicitly exploratory development diagnostics; individual samples are never selected by feature scores",
    }
    save_json(out / "protocol_before_census.json", plan)
    annotations = json.loads(ANNOTATIONS.read_text())
    candidates, split_audits = [], {}
    for split in ("val", "train"):
        print(f"Caption-first census: {split}, {len(catalog)} action labels", flush=True)
        groups, excluded = collect(split, cfg, protocol, annotations, skip_invalid_captions=True)
        rejected = 0
        for group in groups.values():
            for row in group:
                if row["risk_flags"]:
                    rejected += 1
                else:
                    candidates.append((split, row))
        split_audits[split] = {"before_sequence_filter": {label: len(group) for label, group in groups.items()},
                               "sequence_rejections": rejected, "excluded": excluded}
    candidates.sort(key=lambda pair: (pair[0] != "val", hashlib.sha256(
        f"{protocol['seed']}:{pair[1]['motion_id']}".encode()).hexdigest()))
    chosen, seen, sources, splits = [], set(), {}, {}
    for split, row in candidates:
        original = recording(annotations, row["motion_id"])
        if original in seen:
            continue
        seen.add(original)
        chosen.append(row)
        sources[row["sample_id"]] = original
        splits[row["sample_id"]] = split
    order = {c["name"]: i for i, c in enumerate(catalog)}
    chosen.sort(key=lambda row: (order[row["label"]], hashlib.sha256(
        f"{protocol['seed']}:{row['sample_id']}".encode()).hexdigest()))
    for row in chosen:
        if int(row["caption_votes"]) == 1:
            row["review_note"] = "Single matching caption; other captions contain no competing catalog label; motion label pending review"
    write_rows(out / "candidates.csv", chosen)
    counts = {c["name"]: sum(row["label"] == c["name"] for row in chosen) for c in catalog}
    audit = {"counts": counts, "unique_recordings": len(chosen),
             "manifest_sha256": digest(out / "candidates.csv"),
             "source_path_by_sample_id": sources, "split_by_sample_id": splits,
             "split_counts": dict(Counter(splits.values())), "screening": split_audits,
             "labels_human_reviewed": False, "test_motions_used": False}
    audit["per_class_caption_vote_counts"] = {label: dict(Counter(row["caption_votes"] for row in chosen if row["label"] == label)) for label in counts}
    save_json(out / "selection_audit.json", audit)
    save_json(out / "config_snapshot.json", cfg)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
