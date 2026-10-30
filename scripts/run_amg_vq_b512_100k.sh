#!/usr/bin/env bash
set -eu
# Modified for portable private snapshot; see docs/snapshot-changes.md.
T2M_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$T2M_PROJECT_ROOT" || exit 1
T2M_PYTHON="${T2M_PYTHON:-python}"
run_dir=output/hml3d/VQ_amg_b512_k50_100k
mkdir -p "$run_dir"
exec 9>"$run_dir/launcher.lock"
flock -n 9 || { echo 'AMG VQ full training is already running.' >&2; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
args=(
  --dataname t2m --batch-size 512 --window-size 64
  --warm-up-iter 1000 --total-iter 100000 --lr 2e-4 --lr-scheduler 200000
  --print-iter 200 --eval-iter 1000 --save-iter 1000
  --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3
  --quantizer ema_reset --vq-act relu --loss-vel 0.5 --recons-loss l1_smooth
  --amg-pairs-dir output/amg_prepare/humanml3d_train_ready
  --amg-neighbors 50 --amg-temperature 0.07 --amg-contrastive-weight 0.01
  --out-dir output/hml3d --exp-name VQ_amg_b512_k50_100k
)
if [ -f "$run_dir/training_last.pth" ]; then
  args+=(--resume-pth "$run_dir/training_last.pth")
elif [ -f "$run_dir/run.log" ]; then
  echo 'Existing run has no resumable checkpoint; inspect it before restarting.' >&2
  exit 1
else
  mkdir -p "$run_dir/source_snapshot"
  cp --parents train_vq_constra_100.py utils/amg_checkpoint.py utils/eval_trans.py \
    dataset/dataset_amg_vq.py options/option_vq.py scripts/run_amg_vq_b512_100k.sh "$run_dir/source_snapshot/"
  initial_dir=output/hml3d/VQ_amg_b512_k50_check10000
  "$T2M_PYTHON" - "$initial_dir" <<'CHECK'
import json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads((p/'result_summary.json').read_text()); s=torch.load(p/'training_last.pth',map_location='cpu')
assert r['passed'] and r['main_iterations']==10000
assert s['iter']==10000 and s['args']['batch_size']==512 and s['quantizer_ema']['init']
print('Verified 10000-step source state for continuation',flush=True)
CHECK
  cp "$initial_dir"/net_*.pth "$run_dir/"
  cp "$initial_dir/run.log" "$run_dir/initial_10000_run.log"
  sha256sum "$initial_dir/training_last.pth" > "$run_dir/initial_checkpoint_sha256.txt"
  args+=(--resume-pth "$initial_dir/training_last.pth")
  sha256sum output/amg_prepare/humanml3d_train_ready/train_pairs.jsonl \
    output/amg_prepare/missing_tokens/metadata.json > "$run_dir/input_manifest_sha256.txt"
fi
printf '%q ' "$T2M_PYTHON" -u train_vq_constra_100.py "${args[@]}" >> "$run_dir/commands.log"
printf '\n' >> "$run_dir/commands.log"
printf 'running\n' > "$run_dir/status.txt"
set +e
"$T2M_PYTHON" -u train_vq_constra_100.py "${args[@]}" >> "$run_dir/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
if [ "$code" -eq 0 ]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$T2M_PYTHON" - "$run_dir" > "$run_dir/verification.log" 2>&1 <<'VERIFY'
import json,re,sys,torch
from pathlib import Path
run=Path(sys.argv[1]); state=torch.load(run/'training_last.pth',map_location='cpu')
assert state['iter']==100000 and state['scheduler']['last_epoch']==100000
assert state['args']['batch_size']==512 and state['args']['amg_neighbors']==50
assert all(torch.isfinite(x).all() for x in state['net'].values())
assert state['quantizer_ema']['init']
assert torch.isfinite(state['quantizer_ema']['code_sum']).all() and torch.isfinite(state['quantizer_ema']['code_count']).all()
log=(run/'initial_10000_run.log').read_text()+'\n'+(run/'run.log').read_text()
fids=[{'iteration':int(i),'fid':float(v)} for i,v in re.findall(r'Eva\. Iter (\d+) :, FID\. ([\d.]+)',log)]
assert {v['iteration'] for v in fids} >= set(range(0,100001,1000))
report={'passed':True,'main_iterations':100000,'continued_from':10000,'warmup_iterations':1000,'batch_size':512,'positive_neighbors':50,'negative_neighbors':50,'best_fid':state['best'][0],'best_iteration':state['best'][1],'validation_fid':fids}
(run/'result_summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
VERIFY
  verify_code=$?
  if [ "$verify_code" -eq 0 ]; then
    printf 'completed and verified\n' > "$run_dir/status.txt"
  else
    printf 'training completed; checkpoint verification failed: %s\n' "$verify_code" > "$run_dir/status.txt"
  fi
else
  printf 'stopped with exit code %s; inspect console.log\n' "$code" > "$run_dir/status.txt"
fi
exit "$code"
