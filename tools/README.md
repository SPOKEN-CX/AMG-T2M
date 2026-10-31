# Maintained command-line tools

Run these tools from the repository root. This directory keeps the documented VQ → frozen VQ → GPT workflow, the length branch, evaluation, and general inference. Model implementations are in `src/`; input preparation is in `preprocess/`.

## Training entry points

| Entry point | Why it is retained |
|---|---|
| `train_vq.py` | Baseline VQ training for controlled comparisons |
| `train_amg_vq.py` | TMR-guided contrastive VQ training; its objective differs from baseline VQ |
| `train_gpt.py` | Text-conditioned GPT training using a selected frozen VQ |
| `train_length_est.py` | Cached-feature graph/length-estimator training |

These four trainers perform different tasks. Their shared boilerplate does not make them interchangeable.

## Recorded training launchers

| Launcher | Role |
|---|---|
| `run_amg_vq_b512_check10000.sh` | First VQ stage: batch 512, 10,000 steps |
| `run_amg_vq_b512_100k.sh` | Resume that VQ stage to 100,000 total steps |
| `run_amg_gpt.sh` | One GPT launcher with the documented check/full and extended-batch modes |
| `run_amg_lengths.sh` | Prepare/check length features, then train and evaluate the length model |
| `run_amg_vq_test_once.sh` | Formal VQ test-split evaluation for the recorded 100k model |

The first two VQ launchers are successive stages with different resume preconditions. The evaluation launcher calculates metrics rather than training a model. Full commands and prerequisites are in the [main README](../README.md#maintained-training-workflow).

```bash
bash tools/run_amg_gpt.sh check
bash tools/run_amg_gpt.sh full
# Print arguments without loading resources or starting training:
bash tools/run_amg_gpt.sh full --dry-run
```

The consolidated GPT launcher preserves the previous maintained launcher's modes, experiment names, checkpoints, token-cache names, and resume checks. The name change does not rename existing experiment directories.

## Evaluation and inference

| Tool | Role |
|---|---|
| `eval_vq.py` | VQ reconstruction evaluation |
| `eval_gpt.py` | Text-to-motion generation evaluation |
| `run_local.py` | Skeleton preview, VQ reconstruction, small training-path check, and single/multi-prompt generation |
| `predict_amg_lengths.py` | Text-to-duration prediction, called by duration-controlled generation and the length launcher |
| `generate_amg_length_controlled.py` | Generate using matching VQ/GPT/length checkpoints |

For a multi-prompt demo, repeat `--text` in `run_local.py generate` and supply the checkpoint/resource arguments shown in the main README. This replaces the old separate fixed-architecture demo.

## Required internal checks

`checks/` contains two validators:

- `check_amg_gpt_run.py` is invoked after GPT training. It checks checkpoint completion, optimizer/scheduler state, token-cache properties, and frozen VQ identity.
- `check_amg_length_features.py` is invoked by the length pipeline. It checks cached features, graph/text alignment, and cache identity before training.

These are runtime prerequisites of the retained launchers. They have been moved out of the top-level tool list; calls and source-snapshot paths are updated accordingly. `__init__.py` files define the package structure.

## Removed or consolidated entries

| Former entry | Decision |
|---|---|
| `audit_amg_tmr.py` | Removed: date/run-specific one-off audit with obsolete fixed paths. Input preparation already writes its own audit. |
| `check_amg_vq_smoke.py` | Removed: validates only an old 100-step experiment and is not used by the maintained VQ stages. |
| `demo.py` | Removed: fixed-architecture demonstration covered by the general inference utility. |
| `run_amg_vq_check100.sh` | Removed: old startup experiment preset. |
| `run_amg_vq_b1024_check30000.sh` | Removed: alternate batch-1024/30k experiment preset outside the documented primary workflow. |
| `run_amg_vq_full.sh` | Removed: historical batch-128/300k VQ preset. |
| Old `run_amg_gpt.sh` implementation | Replaced by the maintained multi-mode launcher previously named `run_amg_gpt_b512.sh`. |
| `run_amg_gpt_b512.sh` filename | Consolidated into `run_amg_gpt.sh`. |
| `run_gpt_base_a100.sh` | Removed: hardware/run-specific wrapper; the generic GPT trainer remains. |
| `run_gpt_a100_test.sh` | Removed: hardware/run-specific evaluation wrapper; the generic evaluator remains. |

Removed launcher presets used existing trainers with fixed arguments; they were not independent model implementations. Their exact historical contents remain in Git history. This cleanup does not rerun training or remeasure metrics.
