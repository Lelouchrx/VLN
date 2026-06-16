#!/usr/bin/env bash
# Sweep freezen FREEZE_N over {2,3,5,7,8} (skip 4,6), full R2R 1839 pass@1.
# freezen = v3_front4uni generalized: front-N frozen (re-anchor every FREEZE_PERIOD,
# stable cache prefix -> prefill skipped in between) + back (8-N) uniform each round.
# Records SR + mean infer_time_s; prefix-cache hit rate read from the vLLM server log.
set -u
cd /root/code/VLN
ROOT=${ROOT:-/wuji-vefps-D/wuji-il/caiwr/results/freezen_sweep_pass4}
P=${FREEZE_PERIOD:-3}
mkdir -p "$ROOT"
echo "freezen sweep: N in 2 3 5 7 8 (skip 4,6), period=$P, full 1839 pass@1 -> $ROOT"

for N in 2 3 5 7 8; do
    SAVE="$ROOT/N${N}"; rm -rf "$SAVE"
    echo "======== FREEZE_N=$N  start=$(date '+%H:%M:%S') ========"
    s=$(date +%s)
    SAMPLE_STRATEGY=freezen FREEZE_N=$N FREEZE_PERIOD=$P \
        CUDA_VISIBLE_DEVICES=0,1,2,3 RENDER_GPUS=0,1,2,3 CHUNKS=16 PASS_K=4 \
        SAVE_PATH="$SAVE" bash /root/code/VLN/scripts/eval_vllm.sh
    echo "======== FREEZE_N=$N  done in $(( $(date +%s) - s ))s ========"
done

echo "ALL_DONE $(date '+%H:%M:%S')"
echo "=== summary: N  SR  mean_infer_s ==="
for N in 2 3 5 7 8; do
    python3 -c "
import json
f='$ROOT/N${N}/result.json'
s=n=0; t=tn=0
for l in open(f):
    l=l.strip()
    if not l or 'sucs_all' in l: continue
    try: d=json.loads(l)
    except: continue
    if 'success' in d and 'episode_instruction' in d:
        s+=float(d['success']); n+=1
        if d.get('infer_time_s'): t+=float(d['infer_time_s']); tn+=1
print(f'N=$N  SR={s/n:.4f}  mean_infer_s={t/tn:.2f}' if n else 'N=$N no data')
" 2>/dev/null
done
