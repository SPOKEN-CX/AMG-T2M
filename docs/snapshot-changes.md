**Snapshot preparation changes — 2026-10-30**

Only the upload candidate was edited. The original research directory, running experiments, datasets and weights were not modified.

- Selected Python source, necessary configurations and license texts; excluded data, weights, virtual environments, logs, paper files and experiment outputs.
- Excluded unused LibMTL and SMPL mesh rendering pending rights review, plus misleading/incomplete legacy training entry points and destructive download scripts. Kept the skeleton renderer needed by evaluation.
- Changed shell launchers to derive the project root from their location and use `T2M_PYTHON` instead of a personal interpreter path. Hyperparameters and run/resume guards were preserved.
- Changed the legacy length model spaCy path to `T2M_SPACY_MODEL` or `en_core_web_sm`; removed the launcher injection of packages under output. This does not claim the unsupported legacy trainer now works; that trainer is excluded.
- Changed the t-SNE launcher to find the selected Python through command lookup, supporting an environment interpreter name as well as a full path.
- Replaced personal paths in the retained TMR config with relative resource locations. The model architecture and model hyperparameters were preserved.
- Removed TMR text encoder's unconditional Hugging Face mirror override; the caller controls its own endpoint environment.
- Preserved third-party copyright notices and added missing upstream license texts with explicit scope. Exact pre-existing source lineage remains an open item.
- Excluded unused TMR `guofeats/` helpers that require omitted example-motion assets; retained the PyTorch3D-derived geometry helper with its BSD license text.
- Replaced the inherited README and obsolete environment export with a scope-accurate README and observed dependency pins. A fresh installation is pending.

Validation for this candidate is recorded in a separate `packaging-checks.json` document after checks complete. Syntax and CPU regression checks do not establish full model or paper reproduction. The raw security reports, source checksums, tool downloads and preparation scripts stay outside this repository.
