#!/bin/bash
# One server lifetime: install kernel variant, start vllm serve, warm up, greedy-parity probe, bench sweep.
# usage: run_round.sh VARIANT(flash|shipped|guarded) TAG
set -u
VARIANT=$1; TAG=$2
W=${W:-$PWD}; MODEL=/root/model; OUT=$W/results/$TAG
mkdir -p $OUT/telemetry
SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
TARGET=$SITE/v1/attention/ops/triton_unified_attention.py
case $VARIANT in
  flash)   cp $W/kernels/shipped_tel.py $TARGET; BACKEND=FLASH_ATTN ;;
  shipped) cp $W/kernels/shipped_tel.py $TARGET; BACKEND=TRITON_ATTN ;;
  guarded) cp $W/kernels/guarded_tel.py $TARGET; BACKEND=TRITON_ATTN ;;
esac
sha256sum $TARGET > $OUT/kernel-sha256.txt
echo "$(date -u +%FT%TZ) start $TAG variant=$VARIANT backend=$BACKEND" | tee -a $W/results/progress.log
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv > $OUT/nvidia-smi-start.csv
ATTN_TELEMETRY_DIR=$OUT/telemetry VLLM_ATTENTION_BACKEND= vllm serve $MODEL --served-model-name qwen3-8b-awq \
  --dtype bfloat16 --max-model-len 2048 --attention-backend $BACKEND --seed 0 --port 8000 \
  > $OUT/server.log 2>&1 &
SPID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -50 $OUT/server.log; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "server not healthy"; tail -50 $OUT/server.log; kill $SPID; exit 1; }
echo "$(date -u +%FT%TZ) ready $TAG" | tee -a $W/results/progress.log
if [ $VARIANT = guarded ] && ! grep -qh "capture|3d=1|seqs=1|maxk<=2048|tile=64" $OUT/telemetry/*.json; then
  echo "$(date -u +%FT%TZ) GUARD DID NOT FIRE $TAG" | tee -a $W/results/progress.log; kill -INT $SPID; sleep 20; pkill -9 -f "vllm serve"; exit 1; fi
B="vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model qwen3-8b-awq --tokenizer $MODEL \
  --dataset-name random --random-input-len 900 --random-output-len 192 --random-range-ratio 0.85 --ignore-eos \
  --request-rate inf --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99"
# warmup (not saved)
$B --num-prompts 48 --max-concurrency 8 --seed 1 > $OUT/warmup.log 2>&1
python3 $W/parity.py $MODEL $OUT/parity.json > $OUT/parity.log 2>&1
for spec in ${SPECS:-1:64 4:160 8:96 16:160 max:320}; do
  C=${spec%%:*}; N=${spec##*:}
  CC=""; [ "$C" != max ] && CC="--max-concurrency $C"
  $B --num-prompts $N $CC --seed 0 --save-result --save-detailed --result-dir $OUT --result-filename c$C.json \
    > $OUT/bench-c$C.log 2>&1 || echo "bench c$C failed" | tee -a $W/results/progress.log
  echo "$(date -u +%FT%TZ) done $TAG c$C" | tee -a $W/results/progress.log
done
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv > $OUT/nvidia-smi-end.csv
kill -INT $SPID; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null; sleep 5
echo "$(date -u +%FT%TZ) end $TAG" | tee -a $W/results/progress.log
