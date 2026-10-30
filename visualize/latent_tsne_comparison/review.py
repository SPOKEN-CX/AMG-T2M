"""Generate a self-contained, model-blind skeleton review page for the selected motions."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import PROJECT, REPO, digest, read_rows, repo_path, validate_rows


def generate_review(cfg, manifest, out, require_balance=True):
    rows = read_rows(manifest)
    validate_rows(rows, cfg, require_balance=require_balance)
    sys.path.insert(0, str(REPO))
    from utils.motion_process import recover_from_ric
    root = repo_path(cfg["dataset_root"]) / "new_joint_vecs"
    samples = []
    for row in rows:
        raw = np.load(root / f"{row['motion_id']}.npy", allow_pickle=False)
        # Raw new_joint_vecs are not normalized; no inverse transform is needed here.
        with torch.inference_mode():
            joints = recover_from_ric(torch.from_numpy(raw.astype(np.float32)), 22).numpy()
        if not np.isfinite(joints).all():
            raise ValueError("Nonfinite reconstructed joints")
        samples.append({"row": row, "joints": np.round(joints, 3).tolist()})
    payload = {"manifest_sha256": digest(manifest), "fps": cfg["fps"], "window_size": cfg["window_size"],
               "classes": cfg["classes"], "samples": samples}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = (PROJECT / "review_template.html").read_text(encoding="utf-8")
    path = Path(out) / "review.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__REVIEW_DATA__", serialized), encoding="utf-8")
    print(f"Review page: {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
    return path
