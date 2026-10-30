"""Paths, provenance and strict manifest validation shared by the local project."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent.parent
FIELDS = ["sample_id", "motion_id", "source_id", "label", "start_frame", "end_frame",
          "motion_frames", "annotation_start", "annotation_end", "caption", "captions_json",
          "caption_votes", "risk_flags", "review_status", "review_note", "motion_sha256"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path=None):
    path = Path(path or PROJECT / "config.json").resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    names = [c["name"] for c in cfg["classes"]]
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Need at least two distinct classes")
    if cfg["split"] != "test":
        raise ValueError("This experiment is fixed to the held-out test split")
    if cfg["window_size"] < 1 or cfg["samples_per_class"] < 2:
        raise ValueError("Invalid window size or sample count")
    arch = cfg["architecture"]
    if cfg["window_size"] % (arch["stride_t"] ** arch["down_t"]):
        raise ValueError("Window must be divisible by encoder downsampling")
    if len(cfg["models"]) != 2 or len({m["name"] for m in cfg["models"]}) != 2:
        raise ValueError("Exactly two distinct models are required")
    return cfg


def repo_path(value):
    p = Path(value)
    return p if p.is_absolute() else REPO / p


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest is missing required columns: {path}")
        return [{key: row[key] for key in FIELDS} for row in reader]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def source_id(ident):
    return ident[1:] if ident.startswith("M") else ident


def validate_rows(rows, cfg, *, require_reviewed=False, check_files=True, require_balance=True):
    """Prevent split leakage, duplicate/mirrored sources, shifted windows and stale inputs."""
    if not rows:
        raise ValueError("Empty sample manifest")
    root = repo_path(cfg["dataset_root"])
    split = cfg.get("split", "test")
    if split != "test" and not (split == "val" and cfg.get("purpose") == "development_diagnostics"):
        raise ValueError("Validation manifests require an explicit development_diagnostics purpose")
    held_ids = set((root / f"{split}.txt").read_text().split())
    others = {source_id(i) for other in ("train", "val", "test") if other != split
              for i in (root / f"{other}.txt").read_text().split()}
    names = {c["name"] for c in cfg["classes"]}
    seen_sources, seen_samples = set(), set()
    for row in rows:
        ident = row["motion_id"]
        if not re.fullmatch(r"M?\d{6}", ident) or ident not in held_ids:
            raise ValueError(f"Motion is not a {split} ID: {ident}")
        canonical = source_id(ident)
        if canonical in others:
            raise ValueError(f"Train/val mirror overlap: {ident}")
        if row["source_id"] != canonical or canonical in seen_sources:
            raise ValueError(f"Duplicate or invalid canonical motion ID: {ident}")
        seen_sources.add(canonical)
        if row["sample_id"] in seen_samples:
            raise ValueError("Duplicate sample ID")
        seen_samples.add(row["sample_id"])
        if row["label"] not in names:
            raise ValueError(f"Unknown label: {row['label']}")
        if row["review_status"] not in ("pending", "accepted", "rejected", "uncertain"):
            raise ValueError("Invalid review status")
        if row["review_status"] == "rejected":
            raise ValueError("Rejected samples must be removed before extraction")
        if require_reviewed and row["review_status"] != "accepted":
            raise ValueError(f"Sample has not been accepted: {row['sample_id']}")
        start, end = int(row["start_frame"]), int(row["end_frame"])
        lo, hi = int(row["annotation_start"]), int(row["annotation_end"])
        length = int(row["motion_frames"])
        if not (0 <= lo <= start < end <= hi <= length) or end - start != cfg["window_size"]:
            raise ValueError(f"Invalid crop or annotation bounds: {ident}")
        if check_files:
            path = root / "new_joint_vecs" / f"{ident}.npy"
            x = np.load(path, allow_pickle=False)
            if x.shape != (length, cfg["architecture"]["input_dim"]) or not np.isfinite(x).all():
                raise ValueError(f"Invalid or changed motion array: {ident}")
            if digest(path) != row["motion_sha256"]:
                raise ValueError(f"Motion changed after selection: {ident}")
    counts = {name: sum(r["label"] == name for r in rows) for name in names}
    if (require_balance and len(set(counts.values())) != 1) or min(counts.values()) < 2:
        raise ValueError(f"Need balanced classes with at least 2 samples each: {counts}")
    return counts


def versions():
    import sys
    import matplotlib
    import scipy
    import sklearn
    import torch
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}
