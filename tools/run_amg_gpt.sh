#!/usr/bin/env bash
set -euo pipefail
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
mode=${1:-check}
accum=1
validation_limit=0
case "$mode" in
  check) experiment=GPT_amg_b512_check100_bf16; token_name=VQ_amg_b512_tokens_check100; batch=32; total=100; print_every=10; eval_every=50; save_every=50 ;;
  full) experiment=GPT_amg_b512_full_bf16; token_name=VQ_amg_b512_tokens_full; batch=128; total=100000; print_every=200; eval_every=10000; save_every=1000 ;;
  extend300k) experiment=GPT_amg_b512_full_bf16; token_name=VQ_amg_b512_tokens_full; batch=128; total=300000; print_every=200; eval_every=10000; save_every=1000 ;;
  gpt512300k) experiment=GPT_amg_vq512_gpt512_300k_bf16; token_name=VQ_amg_vq512_gpt512_300k_tokens; batch=512; total=300000; print_every=200; eval_every=10000; save_every=1000 ;;
  gpt4096check) experiment=GPT_amg_vq512_gpt4096_check2_bf16; token_name=VQ_amg_vq512_gpt4096_check2_tokens; batch=512; accum=8; total=2; print_every=1; eval_every=2; save_every=2; validation_limit=10 ;;
  gpt4096full) experiment=GPT_amg_vq512_gpt4096_150k_bf16; token_name=VQ_amg_vq512_gpt4096_150k_tokens; batch=512; accum=8; total=150000; print_every=200; eval_every=10000; save_every=1000 ;;
  *) echo 'Usage: bash tools/run_amg_gpt.sh check|full|extend300k|gpt512300k|gpt4096check|gpt4096full [--dry-run]' >&2; exit 2 ;;
esac
if [[ $# -gt 2 || (${2:-} != '' && ${2:-} != --dry-run) ]]; then
  echo 'Unexpected argument' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run_dir=output/hml3d/$experiment
vq=output/hml3d/VQ_amg_b512_frozen/net_best_fid.pth
args=(
  --dataname t2m --exp-name "$experiment" --out-dir output/hml3d
  --batch-size "$batch" --grad-accum-steps "$accum" --total-iter "$total" --lr 1e-4 --lr-scheduler 150000
  --num-layers 9 --embed-dim-gpt 1024 --n-head-gpt 16
  --block-size 51 --ff-rate 4 --drop-out-rate 0.1
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3
  --quantizer ema_reset --vq-act relu --pkeep 0.5
  --eval-iter "$eval_every" --print-iter "$print_every" --save-iter "$save_every" --validation-max-batches "$validation_limit"
  --bf16 --skip-initial-eval --resume-pth "$vq" --vq-name "$token_name"
)
if [[ ${2:-} == --dry-run ]]; then
  printf '%q ' "$T2M_PYTHON" -u tools/train_gpt.py "${args[@]}"
  printf '\n'
  exit 0
fi
[[ -f "$vq" && -f ViT-B-32.pt ]] || { echo 'Frozen VQ or local CLIP weights missing' >&2; exit 1; }
mkdir -p "$run_dir"
exec 9>output/hml3d/GPT_amg_gpu.lock
flock -n 9 || { echo 'Another AMG GPT stage is running' >&2; exit 1; }
"$T2M_PYTHON" - "$mode" "$run_dir" <<'PY'
import json,sys
from pathlib import Path
from preprocess.prepare_amg_gpt import sha256
mode, run_name = sys.argv[1:]
run = Path(run_name)
frozen = json.loads(Path('output/hml3d/VQ_amg_b512_frozen/manifest.json').read_text())
if sha256(frozen['frozen']) != frozen['vq_sha256']:
    raise RuntimeError('Frozen VQ checksum changed')
if mode != 'check':
    report = json.loads(Path('output/hml3d/GPT_amg_b512_check100_bf16/verification.json').read_text())
    if not report['passed'] or report['iteration'] != 100 or report['vq_sha256'] != frozen['vq_sha256']:
        raise RuntimeError('Run and verify the 100-step AMG GPT check first')
if mode == 'extend300k':
    if not (run / 'training_last.pth').is_file() or not (run / 'archive_100000/archive_complete.json').is_file():
        raise RuntimeError('Continuation requires a full training checkpoint and the completed 100k archive')
if mode == 'gpt4096full':
    report = json.loads(Path('output/hml3d/GPT_amg_vq512_gpt4096_check2_bf16/verification.json').read_text())
    if (not report['passed'] or report['iteration'] != 2 or report['vq_sha256'] != frozen['vq_sha256']
            or report['microbatch_size'] != 512 or report['grad_accum_steps'] != 8
            or report['effective_batch_size'] != 4096):
        raise RuntimeError('The 4096-effective-batch smoke check must pass first')
identity = run / 'vq_identity.json'
if identity.exists():
    if json.loads(identity.read_text())['vq_sha256'] != frozen['vq_sha256']:
        raise RuntimeError('Experiment VQ identity differs')
else:
    identity.write_text(json.dumps(frozen, indent=2))
PY
resume=()
if [[ -f "$run_dir/training_last.pth" ]]; then
  resume=(--resume-training "$run_dir/training_last.pth")
elif [[ -f "$run_dir/run.log" ]]; then
  echo 'Existing run has no full checkpoint; inspect before restarting.' >&2
  exit 1
else
  token_dir=dataset/HumanML3D/$token_name
  if [[ -d "$token_dir" ]] && find "$token_dir" -mindepth 1 -print -quit | read -r; then
    echo 'Token directory already contains data without a resumable experiment; inspect before reusing.' >&2
    exit 1
  fi
  mkdir -p "$run_dir/source_snapshot"
  cp --parents tools/train_gpt.py src/options/option_transformer.py src/datasets/dataset_tokenize.py \
    src/datasets/dataset_TM_train.py src/models/t2m_trans.py src/utils/eval_trans.py \
    tools/checks/check_amg_gpt_run.py tools/run_amg_gpt.sh "$run_dir/source_snapshot/"
fi
if [[ "$mode" == extend300k ]]; then
  mkdir -p "$run_dir/continuation_300000_source"
  cp --parents tools/train_gpt.py src/options/option_transformer.py src/datasets/dataset_tokenize.py \
    src/datasets/dataset_TM_train.py tools/run_amg_gpt.sh "$run_dir/continuation_300000_source/"
fi
printf '%q ' "$T2M_PYTHON" -u tools/train_gpt.py "${args[@]}" "${resume[@]}" >> "$run_dir/commands.log"
printf '\n' >> "$run_dir/commands.log"
printf 'running\n' > "$run_dir/status.txt"
set +e
"$T2M_PYTHON" -u tools/train_gpt.py "${args[@]}" "${resume[@]}" >> "$run_dir/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
if [[ "$code" -eq 0 ]]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$T2M_PYTHON" -u \
    tools/checks/check_amg_gpt_run.py --run-dir "$run_dir" --expected-iter "$total" >> "$run_dir/verification.log" 2>&1
  verify_code=$?
  if [[ "$verify_code" -eq 0 ]]; then
    printf 'completed and verified\n' > "$run_dir/status.txt"
  else
    printf 'training completed; verification failed (exit %s)\n' "$verify_code" > "$run_dir/status.txt"
  fi
  exit "$verify_code"
fi
printf 'training stopped with exit code %s; inspect console.log\n' "$code" > "$run_dir/status.txt"
exit "$code"
