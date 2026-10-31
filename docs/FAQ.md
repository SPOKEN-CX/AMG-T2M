# FAQ

## Where do I run commands from?

Run Python commands from the repository root, for example `python tools/train_vq.py ...`. The shell launchers in `tools/` locate the repository root themselves.

## Where should HumanML3D and checkpoints go?

Keep HumanML3D under `dataset/HumanML3D/`; place checkpoints and normalization statistics in the locations listed in [data-and-models.md](data-and-models.md). These resources are excluded from Git.

## Why are experiment settings in shell launchers instead of YAML?

The existing training programs read command-line flags, not YAML. The checked-in `tools/run_*.sh` files therefore remain the executable record of their parameters. `configs/README.md` records this choice; it avoids creating configuration files that do not affect a run.

## How are the exact dependencies installed?

Create the Python interpreter with `environment.yml`, then install the single root `requirements.txt`. It covers VQ/GPT, TMR, length prediction, analysis, and rendering Python dependencies. See [environment.md](environment.md) for system resources and historical dependency records.

## Where are paper authors and DOI?

The paper has not been publicly released. `CITATION.cff` identifies the software with a project-team entity and does not invent manuscript authors, a DOI, or publication details. Update it when those details are approved for release.

## Are all files covered by the root LICENSE?

No. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for retained upstream notices and license scope. The root Apache-2.0 file does not override third-party terms.
