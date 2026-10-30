#!/usr/bin/env python3
"""Independent entry point. All default outputs stay inside this project."""
import argparse
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

from common import PROJECT, digest, load_config, read_rows, save_json, validate_rows


def main():
    parser = argparse.ArgumentParser(description="Paired VQ-VAE latent-space comparison; no training")
    parser.add_argument("command", choices=["prepare", "review", "extract", "analyze", "render", "all", "finalize", "verify"])
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/draft")
    parser.add_argument("--manifest", type=Path, help="Existing sample CSV; default OUTPUT/samples.csv")
    parser.add_argument("--reviewed-csv", type=Path, help="CSV exported by the review page (finalize)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size, unrelated to training batch512")
    parser.add_argument("--require-reviewed", action="store_true", help="Reject pending or uncertain labels")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Only compute the predeclared main t-SNE setting")
    parser.add_argument("--resume", action="store_true", help="Reuse validated t-SNE coordinate files after an interruption")
    parser.add_argument("--force", action="store_true", help="Replace a draft sample selection during prepare")
    args = parser.parse_args()
    cfg, out = load_config(args.config), args.output.resolve()
    manifest = (args.manifest or out / "samples.csv").resolve()
    command = args.command
    if command == "finalize":
        if args.reviewed_csv is None:
            parser.error("finalize requires --reviewed-csv")
        from prepare import finalize
        finalize(cfg, args.reviewed_csv, out)
        return
    if command == "prepare" or (command == "all" and not manifest.exists() and args.manifest is None):
        from prepare import prepare
        if args.require_reviewed:
            parser.error("Cannot create human-reviewed labels automatically; supply a reviewed manifest")
        manifest = prepare(cfg, out, force=args.force)
        if command == "prepare":
            return
    rows = read_rows(manifest)
    validate_rows(rows, cfg, require_reviewed=args.require_reviewed)
    if command in ("all", "review"):
        from review import generate_review
        generate_review(cfg, manifest, out)
    if command in ("all", "extract"):
        from features import extract
        extract(cfg, manifest, out / "features", args.device, args.batch_size, args.require_reviewed)
    if command in ("all", "analyze"):
        from analyze import analyze
        analyze(cfg, manifest, out / "features", out / "results", sensitivity=not args.skip_sensitivity, resume=args.resume)
    if command == "render":
        from analyze import render_saved
        render_saved(cfg, manifest, out / "results")
    if command == "verify":
        from features import verify_parity
        result = verify_parity(cfg, manifest)
        save_json(out / "verification" / "encoder_parity.json", {"manifest_sha256": digest(manifest), "results": result})
        print(result, flush=True)
    if command == "all":
        print(f"Completed. Main figure: {out / 'results/comparison_post.png'}", flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main()
