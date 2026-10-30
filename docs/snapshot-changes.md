**Snapshot preparation changes — 2026-10-30**

Only the upload candidate was edited. The original research directory, running experiments, datasets and weights were not modified.

- Selected Python source, necessary configurations and license texts; excluded data, weights, virtual environments, logs, paper files and experiment outputs.
- Excluded unused LibMTL and SMPL mesh rendering pending rights review, plus misleading/incomplete legacy training entry points and destructive download scripts. Kept the skeleton renderer needed by evaluation.
- Changed shell launchers to derive the project root from their location and use `T2M_PYTHON` instead of a personal interpreter path. Hyperparameters and run/resume guards were preserved.
- Changed the legacy length model spaCy path to `T2M_SPACY_MODEL` or `en_core_web_sm`. The length launcher preserves dependency-directory precedence through the configurable `T2M_LENGTH_DEPS`, defaulting to a portable `.runtime/length_deps` location. The unsupported legacy trainer is excluded.
- Changed the t-SNE launcher to find the selected Python through command lookup, supporting an environment interpreter name as well as a full path.
- Replaced personal paths in the retained TMR config with relative resource locations. The model architecture and model hyperparameters were preserved.
- Removed TMR text encoder's unconditional Hugging Face mirror override; the caller controls its own endpoint environment.
- Preserved third-party copyright notices and added missing upstream license texts with explicit scope. Exact pre-existing source lineage remains an open item.
- Excluded unused TMR `guofeats/` helpers that require omitted example-motion assets; retained the PyTorch3D-derived geometry helper with its BSD license text.
- Replaced the inherited README and obsolete environment export with a scope-accurate README and exact package records: 105 main-environment packages, 4 TMR overlay packages and 50 length overlay packages. Python is fixed at 3.10.18 and the recorded PyTorch runtime build is 2.0.1+cu117. Main and overlay versions retain their original loading precedence.

Validation for this candidate is recorded in a separate `packaging-checks.json` document after checks complete. Syntax and CPU regression checks do not establish full model or paper reproduction. The raw security reports, source checksums, tool downloads and preparation scripts stay outside this repository.

At the user's request, fresh-environment installation and demo validation were deferred. The CPU demo and quantizer portability work were preserved outside the repository and are not part of this update. No model implementation or training hyperparameter change is included in the dependency-recording update. Package declaration constraints were checked against the observed base and overlay metadata; this is a static dependency audit.

The repository documentation now includes a 38-entry external-resource catalog, source-based workflow instructions, the VQ/GPT batch and resume relationships, and issue forms. The paper remains unpublished by user confirmation; only its title is retained, and no paper-author citation, public release or DOI was added. The CFF file identifies the software using a generic project-team entity.

**Architecture update — 2026-10-31**

The private `main` branch was reorganized into `src/`, `tools/`, `preprocess/`, `visualize/`, `configs/`, `assets/`, and `docs/`. Entrypoint imports and repository-relative dataset roots were updated; external data, model weights, caches, outputs, and exact dependency lock files were left out of the move. The current experiment parameters remain in executable `tools/run_*.sh` launchers because the training programs do not read project YAML. A Python 3.10.18 Dockerfile and `.dockerignore` were added but not built or runtime-validated, as requested.
