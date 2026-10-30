# Experiment configuration

Current training entry points take their hyperparameters from command-line arguments. The version-controlled `tools/run_*.sh` launchers are the executable source of truth for the recorded runs; this directory does not contain duplicate YAML that the programs would ignore.

When the training entry points gain YAML loading, add one file per experiment here and keep each launcher thin. The retained TMR upstream Hydra configuration stays under `src/models/retrieval/configs/` so its relative config lookup remains intact.
