#!/usr/bin/env bash
set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/eight-direction-screen/logs
pids=()
workers=${SLURM_CPUS_PER_TASK:-4}
for ((start=0; start<workers; start++)); do
 .pixi/envs/default/bin/python scripts/screen_eight_directions.py \
   --start "$start" --stride "$workers" --chunk-size 2 \
   > "results/eight-direction-screen/logs/worker-${start}.log" 2>&1 &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
