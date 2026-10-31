# Local reproduction source

This directory preserves the local project's original model, loader, training, evaluation, feature-processing, and rendering layout. Use it as the source baseline for local experimental variants. Run its commands with this directory as the working directory so that `models`, `dataset`, `utils`, and the nested TMR `src` package resolve consistently.

## Install once, from the repository root

```bash
conda env create -f environment.yml
conda activate AMG-T2M
python -m pip install -r requirements.txt
```

There is one active requirements file at the repository root. It includes TMR and length prediction; no extra nested requirements file or `.runtime` installation is needed. See the [environment guide](../../docs/environment.md). The old dependency specifications are preserved as exact-text records in [dependency-history.json](../../docs/dependency-history.json).

## Configure this source copy

From the repository root:

```bash
cd reproduction/local
export PYTHONPATH="$PWD:$PWD/models/Retrival${PYTHONPATH:+:$PYTHONPATH}"

# Inspect path substitutions first.
python ../configure_paths.py --data-root /path/to/Dataset
# Apply them to this checkout's source copy.
python ../configure_paths.py --data-root /path/to/Dataset --spacy-model en_core_web_sm --write
```

`--data-root` is the parent containing `HumanML3D/`, `KIT-ML/`, or `BABEL_TEACH/`. The helper changes known resource-path strings in manifest-listed Python/JSON files. It does not download or modify datasets, change model computations, or tune training parameters. Running it changes the published source hashes locally, as expected.

Checkpoint resources, GloVe, CLIP and normalization files must be placed under this working tree's original relative locations. See the repository's [English README](../../README.md#data-and-external-resources) for source URLs and the resource layout.

## Entry points

| Task | Local entry points |
|---|---|
| VQ training | `train_vq.py`, `train_vq_constra.py`, `train_vq_constra_100.py` |
| Distributed VQ variants | `train_vq_multi.py`, `train_vq_constra_multi.py` |
| Text-conditioned GPT | `train_t2m_trans.py` |
| Formal model evaluation | `VQ_eval.py`, `GPT_eval_multi.py` |
| Length models | `models/MotionLenEstimator/` |
| TMR encoding/retrieval | `models/Retrival/` |
| Preview/reconstruction/inference | `scripts/run_local.py` |
| Mesh and skeleton rendering | `visualize/`, `visualization/`, `render_final.py` |
| Historical command notes | `run_order` |

The original working guides remain available in Chinese: [training and evaluation](AMG_TRAIN_EVAL_GUIDE.zh-CN.md), [local execution](LOCAL_RUN_GUIDE.zh-CN.md), and [server execution](SERVER_RUN_GUIDE.zh-CN.md). Their historical environment notes are superseded by the shared installation above. The [main English README](../../README.md#maintained-training-workflow) documents maintained root-level commands for the same project components.

The [source audit](../../docs/local-source-sync.md) records historical empty files, duplicated entry points, and references to files absent from the supplied directory. In particular, `train_transition.py` duplicates the GPT trainer, `train_GPT_split.py` is empty, and the old length trainer imports missing `utils/utils.py`. Select executable entry points by their actual code and documented workflow.

## Provenance and resources

The [source manifest](../source-manifest.json) identifies the original 200 local source/configuration/documentation files and their original hashes. Later dependency consolidation is recorded in its change fields; removed dependency files are retained as members of the historical JSON archive. The original upstream project README is available as [README.upstream.md](README.upstream.md).

The `dataset/` directory contains source loaders and historical resource scripts. The `checkpoints/` directory contains only seven text `opt.txt` model configurations. Motion data, captions, splits, indexes, normalization tensors, GloVe, weights, SMPL assets, and generated outputs are external resources. Use the official [dataset sources](../../README.md#data-and-external-resources); retain their applicable terms and the [third-party notices](../../THIRD_PARTY_NOTICES.md).

The historical download scripts contain destructive directory operations and inconsistent paths. They are retained for source provenance; prepare resources through the official instructions referenced by the main README.
