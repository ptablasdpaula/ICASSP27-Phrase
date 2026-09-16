#!/usr/bin/env bash
set -euo pipefail
repo=${1:?repository path}
cd "$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONHOME="$repo/.pixi/envs/default"
exec "$repo/.pixi/envs/default/bin/python" scripts/validate_orthogonal_cases.py \
    --index "${SLURM_ARRAY_TASK_ID:?}"
