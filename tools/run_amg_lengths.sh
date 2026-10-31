#!/usr/bin/env bash
set -euo pipefail
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Dependencies are installed together through the root requirements.txt.
# An explicitly requested legacy overlay remains supported.
if [[ -n "${T2M_LENGTH_DEPS:-}" ]]; then
  [[ -d "$T2M_LENGTH_DEPS" ]] || { echo 'Configured T2M_LENGTH_DEPS directory missing' >&2; exit 1; }
  export PYTHONPATH="$T2M_LENGTH_DEPS${PYTHONPATH:+:$PYTHONPATH}"
fi
python_bin="$T2M_PYTHON"
root=output/amg_prepare/length_pipeline_b512_50k
features=output/amg_prepare/length_features_v1
parser="${T2M_SPACY_MODEL:-en_core_web_sm}"
smoke=output/hml3d/Length_amg_gat_b512_check20
run=output/hml3d/Length_amg_gat_b512_50k
mkdir -p "$root"
exec 9>"$root/pipeline.lock"
flock -n 9 || { echo 'Length pipeline already running' >&2; exit 1; }
exec >>"$root/console.log" 2>&1
trap 'code=$?; printf "stopped with exit %s; inspect console.log\n" "$code" > "$root/status.txt"' ERR
printf 'preparing frozen word features and graphs\n' > "$root/status.txt"
if [[ ! -f "$features/manifest.json" ]]; then
  "$python_bin" -u preprocess/prepare_amg_length_features.py --output "$features" \
    --spacy-model "$parser" --batch-size 128
fi
"$python_bin" -u tools/check_amg_length_features.py --features "$features" --spacy-model "$parser"
printf 'checking actual batch512 training for 20 steps\n' > "$root/status.txt"
if [[ ! -f "$smoke/completion.json" ]]; then
  smoke_resume=()
  [[ ! -f "$smoke/training_last.pth" ]] || smoke_resume=(--resume "$smoke/training_last.pth")
  "$python_bin" -u tools/train_length_est.py --features "$features" --out-dir "$smoke" \
    --batch-size 512 --total-iter 20 --eval-iter 20 --print-iter 10 --skip-test "${smoke_resume[@]}"
fi
"$python_bin" - "$smoke" <<'PY'
from pathlib import Path
import json,sys
run=Path(sys.argv[1])
report=json.loads((run/'completion.json').read_text())
startup=json.loads((run/'startup_check.json').read_text())
assert report['passed'] and report['iteration']==20 and report['batch_size']==512
assert startup['passed'] and startup['actual_batch_size']==512 and startup['first_gradient_norm']>0
PY
snapshot="$run/source_snapshot"
if [[ -f "$snapshot/tools/train_length_est.py" ]]; then
  snapshot="$run/resume_source_snapshot"
fi
mkdir -p "$snapshot"
cp --parents tools/train_length_est.py src/datasets/dataset_amg_lengths.py \
  src/models/motion_length_estimator/cached_length.py src/models/motion_length_estimator/LengthEst.py \
  preprocess/prepare_amg_length_features.py tools/check_amg_length_features.py \
  tools/predict_amg_lengths.py tools/run_amg_lengths.sh "$snapshot/"
resume=()
[[ ! -f "$run/training_last.pth" ]] || resume=(--resume "$run/training_last.pth")
args=(--features "$features" --out-dir "$run" --batch-size 512 --total-iter 50000 \
  --eval-iter 1000 --save-iter 1000 --print-iter 100 --lr 0.001)
printf '%q ' "$python_bin" -u tools/train_length_est.py "${args[@]}" "${resume[@]}" >> "$run/commands.log"
printf '\n' >> "$run/commands.log"
printf 'training length model: batch512, 50000 steps\n' > "$root/status.txt"
printf 'running\n' > "$run/status.txt"
set +e
"$python_bin" -u tools/train_length_est.py "${args[@]}" "${resume[@]}" >> "$run/console.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$run/exit_code.txt"
if [[ "$code" -ne 0 ]]; then
  printf 'stopped with exit %s\n' "$code" > "$run/status.txt"
  exit "$code"
fi
"$python_bin" - "$run" <<'PY'
import json,sys,torch
from pathlib import Path
run=Path(sys.argv[1])
report=json.loads((run/'completion.json').read_text())
test=json.loads((run/'test_metrics.json').read_text())
state=torch.load(run/'training_last.pth',map_location='cpu')
assert report['passed'] and report['iteration']==50000 and report['batch_size']==512 and report['test_completed']
assert state['iter']==50000 and all(torch.isfinite(t).all() for t in state['model'].values())
assert test['samples']>0 and test['mae_frames']>=0
(run/'verification.json').write_text(json.dumps({'passed':True,'iteration':50000,'batch_size':512,'test':test},indent=2))
PY
"$python_bin" -u tools/predict_amg_lengths.py --checkpoint "$run/net_best_mae.pth" \
  --spacy-model "$parser" --text 'A person walks forward.' 'A person squats, stands up, and turns around.' \
  --output "$run/example_predictions.json"
printf 'completed and verified\n' > "$run/status.txt"
printf 'completed and verified; test metrics saved\n' > "$root/status.txt"
