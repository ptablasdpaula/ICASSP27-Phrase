#!/usr/bin/env bash
# Submit with GPU resources; compute mode uses array indices 0–172.
set -euo pipefail
repo=${1:?repository path}
output=${2:?output directory}
mode=${3:?qualify, compute or figure}
cd "$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="$repo/.pixi/envs/default/bin/python"
case "$mode" in
  qualify)
    exec "$python_bin" scripts/screen_cel_gradients.py qualify \
      --output "$output/qualification.json" ;;
  compute)
    exec "$python_bin" scripts/screen_cel_gradients.py compute --device cuda \
      --index "${SLURM_ARRAY_TASK_ID:?}" --qualification "$output/qualification.json" \
      --output "$output/raw" ;;
  figure)
    exec "$python_bin" scripts/recompute_loss_landscapes.py --device cuda \
      --output "$output/loss_landscapes" ;;
  *) exit 2 ;;
esac
