#!/usr/bin/env bash
set -u -o pipefail
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

root=${GPT_EVAL_ROOT:-output/hml3d_eval/GPT_a100_test_managed}
mkdir -p "$root"
sha256sum output/hml3d/VQ_base/net_best_fid.pth output/hml3d/GPT_base_a100_bf16/net_best_fid.pth > "$root/checkpoint_sha256.txt" || exit 1
args=(--dataname t2m --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16
  --block-size 51 --ff-rate 4 --drop-out-rate 0.1
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3
  --quantizer ema_reset --vq-act relu
  --resume-pth output/hml3d/VQ_base/net_best_fid.pth
  --resume-trans output/hml3d/GPT_base_a100_bf16/net_best_fid.pth
  --out-dir "$root")

printf 'single evaluation started\n' > "$root/status.txt"
"$T2M_PYTHON" -u tools/eval_gpt.py "${args[@]}" --exp-name once --repeat-time 1 > "$root/once.console.log" 2>&1
code=$?
if (( code != 0 )); then
  printf 'single evaluation failed: exit %d\n' "$code" > "$root/status.txt"
  exit "$code"
fi
if ! grep -q 'Multimodality.*conf' "$root/once/run.log"; then
  printf 'single evaluation missing summary\n' > "$root/status.txt"
  exit 1
fi
printf 'single evaluation passed; formal 20-repeat evaluation started\n' > "$root/status.txt"
"$T2M_PYTHON" -u tools/eval_gpt.py "${args[@]}" --exp-name formal20 --repeat-time 20 > "$root/formal20.console.log" 2>&1
code=$?
if (( code != 0 )); then
  printf 'formal evaluation failed: exit %d\n' "$code" > "$root/status.txt"
  exit "$code"
fi
if ! grep -q 'Multimodality.*conf' "$root/formal20/run.log"; then
  printf 'formal evaluation missing summary\n' > "$root/status.txt"
  exit 1
fi
printf 'formal 20-repeat evaluation completed successfully\n' > "$root/status.txt"
