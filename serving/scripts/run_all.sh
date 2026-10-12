#!/bin/bash
# ABCCBA: FLASH_ATTN, shipped TRITON_ATTN, guarded tile64, guarded, shipped, FLASH_ATTN.
W=/root/w
for r in "flash r1-flash" "shipped r1-shipped" "guarded r1-guarded" "guarded r2-guarded" "shipped r2-shipped" "flash r2-flash"; do
  set -- $r
  [ -f $W/results/$2/c$( echo max ).json ] && { echo "skip $2 (done)"; continue; }
  bash $W/run_round.sh $1 $2
done
echo "$(date -u +%FT%TZ) ALL DONE" | tee -a $W/results/progress.log
