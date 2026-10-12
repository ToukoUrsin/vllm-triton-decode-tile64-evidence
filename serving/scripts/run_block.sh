#!/bin/bash
# One block = the three variants back to back on one machine (paired). usage: run_block.sh BLOCK "v1 v2 v3"
W=$PWD; B=$1
for v in $2; do
  [ -f $W/results/$B-$v/cmax.json ] && continue
  bash $W/run_round.sh $v $B-$v || echo "$(date -u +%FT%TZ) round $B-$v failed" >> $W/results/progress.log
done
echo "$(date -u +%FT%TZ) ALL DONE $B" | tee -a $W/results/progress.log
