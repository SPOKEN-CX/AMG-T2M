#!/usr/bin/env bash
set -euo pipefail
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
run_dir=output/hml3d/GPT_base_a100_bf16
mkdir -p "$run_dir"
resume_args=()
if [[ -f "$run_dir/training_last.pth" ]]; then
  resume_args=(--resume-training "$run_dir/training_last.pth")
fi
set +e
"$T2M_PYTHON" -u train_t2m_trans.py \
 --dataname t2m --exp-name GPT_base_a100_bf16 --out-dir output/hml3d \
 --batch-size 128 --total-iter 300000 --lr 1e-4 --lr-scheduler 150000 \
 --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16 \
 --block-size 51 --ff-rate 4 --drop-out-rate 0.1 \
 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
 --quantizer ema_reset --vq-act relu --pkeep 0.5 \
 --eval-iter 10000 --print-iter 200 --save-iter 1000 --bf16 --skip-initial-eval \
 --resume-pth output/hml3d/VQ_base/net_best_fid.pth --vq-name VQ_base_tokens \
 "${resume_args[@]}" >> "$run_dir/console.log" 2>&1
training_exit=$?
printf '%s\n' "$training_exit" > "$run_dir/exit_code.txt"
exit "$training_exit"
