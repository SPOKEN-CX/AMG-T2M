"""Load the existing encoders/codebooks only; never load or run the TMR teacher."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import REPO, digest, read_rows, repo_path, save_json, validate_rows, versions

sys.path.insert(0, str(REPO))
from models.encdec import Encoder


class FrozenEncoder(nn.Module):
    """Exact encoder + nearest-codebook lookup, without the unused decoder/CUDA-only constructor."""
    def __init__(self, cfg, checkpoint):
        super().__init__()
        arch = cfg["architecture"]
        state = torch.load(checkpoint, map_location="cpu")["net"]
        self.encoder = Encoder(arch["input_dim"], arch["output_emb_width"], arch["down_t"],
                               arch["stride_t"], arch["width"], arch["depth"],
                               arch["dilation_growth_rate"], activation=arch["activation"], norm=arch["norm"])
        prefix = "vqvae.encoder."
        weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.encoder.load_state_dict(weights, strict=True)
        book = state["vqvae.quantizer.codebook"]
        if tuple(book.shape) != (arch["nb_code"], arch["code_dim"]):
            raise ValueError("Checkpoint codebook shape differs from configuration")
        if any(not torch.isfinite(v).all() for v in weights.values()) or not torch.isfinite(book).all():
            raise ValueError("Checkpoint has nonfinite weights")
        self.register_buffer("codebook", book)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        z = self.encoder(x.permute(0, 2, 1).float())  # B,C,T'
        flat = z.permute(0, 2, 1).contiguous().view(-1, z.shape[1])
        distances = (flat.square().sum(-1, keepdim=True)
                     - 2 * flat @ self.codebook.T + self.codebook.square().sum(-1)[None])
        indices = distances.argmin(-1).view(len(x), -1)
        quantized = torch.nn.functional.embedding(indices, self.codebook)
        return z.mean(-1), quantized.mean(1), indices


def motion_batch(rows, cfg):
    stats = repo_path(cfg["normalization_dir"])
    mean, std = np.load(stats / "mean.npy"), np.load(stats / "std.npy")
    dim = cfg["architecture"]["input_dim"]
    if mean.shape != (dim,) or std.shape != (dim,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid VQ normalization statistics")
    root = repo_path(cfg["dataset_root"]) / "new_joint_vecs"
    result = []
    for r in rows:
        x = np.load(root / f"{r['motion_id']}.npy", allow_pickle=False)
        x = x[int(r["start_frame"]):int(r["end_frame"])]
        result.append(((x - mean) / std).astype(np.float32))
    return np.stack(result)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms <= 1e-12).any():
        raise ValueError("Nonfinite or zero latent representation")
    return x / norms


def extract(cfg, manifest, out, device="auto", batch_size=16, require_reviewed=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = read_rows(manifest)
    counts = validate_rows(rows, cfg, require_reviewed=require_reviewed)
    device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    torch.manual_seed(cfg["selection_seed"])
    torch.set_num_threads(4)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["selection_seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    raw = motion_batch(rows, cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = np.array([r["label"] for r in rows])
    info = {"manifest_sha256": digest(manifest), "config": cfg, "counts": counts,
            "labels_human_accepted": all(r["review_status"] == "accepted" for r in rows),
            "versions": versions(), "device": device, "batch_size": batch_size,
            "feature_definition": "temporal mean of encoder output / nearest codebook embeddings, followed by L2 normalization",
            "normalization_sha256": {n: digest(repo_path(cfg["normalization_dir"]) / n) for n in ("mean.npy", "std.npy")},
            "models": []}
    for spec in cfg["models"]:
        path = repo_path(spec["checkpoint"])
        model = FrozenEncoder(cfg, path).to(device)
        pre, post, tokens = [], [], []
        with torch.inference_mode():
            for i in range(0, len(raw), batch_size):
                ze, zq, idx = model(torch.from_numpy(raw[i:i + batch_size]).to(device))
                pre.append(ze.cpu().numpy())
                post.append(zq.cpu().numpy())
                tokens.append(idx.cpu().numpy())
        pre, post, tokens = np.concatenate(pre), np.concatenate(post), np.concatenate(tokens)
        np.savez_compressed(out / f"{spec['name']}.npz", pre_raw=pre, post_raw=post,
                            pre=normalize(pre), post=normalize(post), tokens=tokens, labels=labels,
                            sample_ids=np.array([r["sample_id"] for r in rows]),
                            source_ids=np.array([r["source_id"] for r in rows]),
                            manifest_sha256=np.array(info["manifest_sha256"]))
        info["models"].append({**spec, "checkpoint_absolute": str(path.resolve()), "checkpoint_sha256": digest(path)})
        print(f"Extracted {spec['name']}: {len(pre)} x {pre.shape[1]}, tokens {tokens.shape}", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    save_json(out / "provenance.json", info)
    return info


def verify_parity(cfg, manifest):
    """Integration check against the repository's complete HumanVQVAE and real weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("Repository quantizer constructor needs CUDA for this parity check; extraction itself supports CPU")
    from argparse import Namespace
    from models.vqvae import HumanVQVAE
    arch = cfg["architecture"]
    args = Namespace(dataname="t2m", quantizer="ema_reset", mu=0.99)
    # Use one real crop from each class so phase and codebook lookup are exercised.
    rows = read_rows(manifest)
    selected = [next(r for r in rows if r["label"] == c["name"]) for c in cfg["classes"]]
    x = torch.from_numpy(motion_batch(selected, cfg)).cuda()
    results = []
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        for spec in cfg["models"]:
            path = repo_path(spec["checkpoint"])
            exact = HumanVQVAE(args, arch["nb_code"], arch["code_dim"], arch["output_emb_width"],
                               arch["down_t"], arch["stride_t"], arch["width"], arch["depth"],
                               arch["dilation_growth_rate"], arch["activation"], arch["norm"]).cuda().eval()
            exact.load_state_dict(torch.load(path, map_location="cpu")["net"], strict=True)
            slim = FrozenEncoder(cfg, path).cuda().eval()
            pre, post, indices = slim(x)
            expected_pre = exact.vqvae.encoder(exact.vqvae.preprocess(x)).mean(-1)
            expected_post = exact(x)[3]
            expected_indices = exact.encode(x)
            torch.testing.assert_close(pre, expected_pre, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(post, expected_post, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
            results.append({"model": spec["name"], "samples": len(x),
                            "pre_max_abs_error": float((pre - expected_pre).abs().max()),
                            "post_max_abs_error": float((post - expected_post).abs().max()),
                            "tokens_identical": True})
            del exact, slim
            torch.cuda.empty_cache()
    return results
