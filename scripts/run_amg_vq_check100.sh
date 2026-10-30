#!/usr/bin/env bash
set -u
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
run_dir=output/hml3d/VQ_amg_check100
if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite existing experiment: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir" || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$T2M_PYTHON" -u train_vq_constra_100.py \
  --dataname t2m --batch-size 32 --window-size 64 \
  --warm-up-iter 10 --total-iter 100 --print-iter 10 --eval-iter 50 --save-iter 50 \
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
  --quantizer ema_reset --vq-act relu --loss-vel 0.5 --recons-loss l1_smooth \
  --amg-pairs-dir output/amg_prepare/humanml3d_train_ready \
  --amg-neighbors 10 --amg-temperature 0.07 --amg-contrastive-weight 0.01 \
  --out-dir output/hml3d --exp-name VQ_amg_check100 \
  > "$run_dir/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
exit "$code"
