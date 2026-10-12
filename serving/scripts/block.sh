#!/bin/bash
# fleet-run entry: setup, then one block of three variants in the given order. Resumable after spot eviction.
cd "$(dirname "$0")"; mkdir -p results
bash setup.sh > results/setup-$FLEET_RUN_ATTEMPT.log 2>&1 || { tail -20 results/setup-$FLEET_RUN_ATTEMPT.log; exit 1; }
bash run_block.sh "$1" "$2"
tail -20 results/progress.log
