#!/usr/bin/env bash
# Stage 2: after the N-sweep finishes, pick the best N (by SR), then sweep the freeze
# re-anchor PERIOD over {2..8}, full R2R 1839 pass@1. Records SR + mean infer_time_s.
set -u
cd /root/code/VLN
ROOT_N=/wuji-vefps-D/wuji-il/caiwr/results/freezen_sweep_pass4
ROOT_P=/wuji-vefps-D/wuji-il/caiwr/results/freezen_period_sweep_pass4
mkdir -p "$ROOT_P"

sr_of(){ python3 -c "
import json
s=n=0
for l in open('$1'):
    l=l.strip()
    if not l or 'sucs_all' in l: continue
    try: d=json.loads(l)
    except: continue
    if 'success' in d and 'episode_instruction' in d: s+=float(d['success']); n+=1
print(f'{s/n:.4f}' if n else '0')
" 2>/dev/null; }

echo "[period] waiting for N-sweep to finish..."
while true; do
    ok=1
    for N in 2 3 5 7 8; do
        tail -n1 "$ROOT_N/N${N}/result.json" 2>/dev/null | grep -q 'sucs_all' || ok=0
    done
    [ "$ok" = 1 ] && break; sleep 60
done

# best N by SR
BESTN=2; BESTSR=0
for N in 2 3 5 7 8; do
    sr=$(sr_of "$ROOT_N/N${N}/result.json")
    echo "[period] N=$N SR=$sr"
    awk "BEGIN{exit !($sr>$BESTSR)}" && { BESTSR=$sr; BESTN=$N; }
done
echo "[period] BEST N = $BESTN (SR $BESTSR); sweeping PERIOD 2..8"

for P in 2 3 4 5 6 7 8; do
    SAVE="$ROOT_P/N${BESTN}_P${P}"; rm -rf "$SAVE"
    echo "======== N=$BESTN PERIOD=$P start=$(date '+%H:%M:%S') ========"
    s=$(date +%s)
    SAMPLE_STRATEGY=freezen FREEZE_N=$BESTN FREEZE_PERIOD=$P \
        CUDA_VISIBLE_DEVICES=0,1,2,3 RENDER_GPUS=0,1,2,3 CHUNKS=16 PASS_K=4 \
        SAVE_PATH="$SAVE" bash /root/code/VLN/scripts/eval_vllm.sh
    echo "======== N=$BESTN PERIOD=$P done in $(( $(date +%s) - s ))s ========"
done

echo "PERIOD_DONE $(date '+%H:%M:%S')  (best N=$BESTN)"
for P in 2 3 4 5 6 7 8; do
    python3 -c "
import json
f='$ROOT_P/N${BESTN}_P${P}/result.json'
s=n=t=tn=0
for l in open(f):
    l=l.strip()
    if not l or 'sucs_all' in l: continue
    try: d=json.loads(l)
    except: continue
    if 'success' in d and 'episode_instruction' in d:
        s+=float(d['success']); n+=1
        if d.get('infer_time_s'): t+=float(d['infer_time_s']); tn+=1
print(f'N=$BESTN P=$P  SR={s/n:.4f}  mean_infer_s={t/tn:.2f}' if n else 'P=$P no data')
" 2>/dev/null
done
