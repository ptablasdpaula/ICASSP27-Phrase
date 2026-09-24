#!/usr/bin/env bash
set -euo pipefail
# sbatch --chdir supplies the checkout, including when Slurm copies this script.
: "${PHRASE_REPO:?Submit through jobs/submit.py}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec pixi run --manifest-path "$PHRASE_REPO/pixi.toml" python "$PHRASE_REPO/jobs/task.py" "$@" --task "${SLURM_ARRAY_TASK_ID:-0}"
