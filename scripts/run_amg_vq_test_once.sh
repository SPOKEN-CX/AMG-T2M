#!/usr/bin/env bash
set -u
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
run_dir=output/hml3d_eval/VQ_amg_b512_test_once
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing evaluation: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir" || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
args=(--dataname t2m --batch-size 32 --nb-code 512 --down-t 2 --depth 3
      --dilation-growth-rate 3 --quantizer ema_reset --vq-act relu
      --resume-pth output/hml3d/VQ_amg_b512_k50_100k/net_best_fid.pth
      --out-dir output/hml3d_eval --exp-name VQ_amg_b512_test_once --repeat-time 1)
printf '%q ' "$T2M_PYTHON" -u VQ_eval.py "${args[@]}" > "$run_dir/command.log"
printf '\n' >> "$run_dir/command.log"
printf 'running\n' > "$run_dir/status.txt"
"$T2M_PYTHON" -u VQ_eval.py "${args[@]}" > "$run_dir/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
if [[ "$code" -eq 0 && -f "$run_dir/metrics.json" ]]; then
  printf 'completed\n' > "$run_dir/status.txt"
else
  printf 'failed: exit %s\n' "$code" > "$run_dir/status.txt"
fi
exit "$code"
