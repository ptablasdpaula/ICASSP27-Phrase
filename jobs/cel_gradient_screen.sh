#!/usr/bin/env bash
# Submit with GPU resources; compute mode uses array indices 0–172.
set -euo pipefail
repo=${1:?repository path}
output=${2:?output directory}
mode=${3:?qualify, compute, cpu or figure}
cd "$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHOME="$repo/.pixi/envs/default"
python_bin="$repo/.pixi/envs/default/bin/python"
case "$mode" in
  qualify)
    exec "$python_bin" scripts/screen_cel_gradients.py qualify \
      --output "$output/qualification.json" ;;
  compute)
    exec "$python_bin" scripts/screen_cel_gradients.py compute --device cuda \
      --index "${SLURM_ARRAY_TASK_ID:?}" --qualification "$output/qualification.json" \
      --output "$output/raw" ;;
  cpu)
    mkdir -p "$output"
    workers=${SLURM_CPUS_PER_TASK:-1}
    pids=()
    for ((index=0; index<workers; index++)); do
      "$python_bin" scripts/screen_cel_gradients.py compute --device cpu \
        --start "$index" --stride "$workers" --chunk-size 4 --output "$output/raw" \
        > "$output/worker-${index}.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    exit "$status" ;;
  figure)
    exec "$python_bin" scripts/recompute_loss_landscapes.py --device cuda \
      --output "$output/loss_landscapes" ;;
  *) exit 2 ;;
esac
