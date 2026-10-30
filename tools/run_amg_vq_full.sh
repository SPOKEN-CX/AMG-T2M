#!/usr/bin/env bash
set -eu
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
run_dir=output/hml3d/VQ_amg_full
mkdir -p "$run_dir"
exec 9>"$run_dir/launcher.lock"
flock -n 9 || { echo 'AMG VQ full training is already running.' >&2; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
args=(
  --dataname t2m --batch-size 128 --window-size 64
  --warm-up-iter 1000 --total-iter 300000 --lr 2e-4 --lr-scheduler 200000
  --print-iter 200 --eval-iter 5000 --save-iter 5000
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3
  --quantizer ema_reset --vq-act relu --loss-vel 0.5 --recons-loss l1_smooth
  --amg-pairs-dir output/amg_prepare/humanml3d_train_ready
  --amg-neighbors 50 --amg-temperature 0.07 --amg-contrastive-weight 0.01
  --out-dir output/hml3d --exp-name VQ_amg_full
)
if [ -f "$run_dir/training_last.pth" ]; then
  args+=(--resume-pth "$run_dir/training_last.pth")
elif [ -f "$run_dir/run.log" ]; then
  echo 'Existing run has no resumable checkpoint; inspect it before restarting.' >&2
  exit 1
else
  mkdir -p "$run_dir/source_snapshot"
  cp --parents tools/train_amg_vq.py src/utils/amg_checkpoint.py src/utils/eval_trans.py \
    src/datasets/dataset_amg_vq.py src/options/option_vq.py tools/run_amg_vq_full.sh "$run_dir/source_snapshot/"
  sha256sum output/amg_prepare/humanml3d_train_ready/train_pairs.jsonl \
    output/amg_prepare/missing_tokens/metadata.json > "$run_dir/input_manifest_sha256.txt"
fi
printf '%q ' "$T2M_PYTHON" -u tools/train_amg_vq.py "${args[@]}" >> "$run_dir/commands.log"
printf '\n' >> "$run_dir/commands.log"
printf 'running\n' > "$run_dir/status.txt"
set +e
"$T2M_PYTHON" -u tools/train_amg_vq.py "${args[@]}" >> "$run_dir/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
if [ "$code" -eq 0 ]; then
  printf 'completed\n' > "$run_dir/status.txt"
else
  printf 'stopped with exit code %s; inspect console.log\n' "$code" > "$run_dir/status.txt"
fi
exit "$code"
