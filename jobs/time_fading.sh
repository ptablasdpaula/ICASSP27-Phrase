#!/usr/bin/env bash
set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .pixi/envs/default/bin/python scripts/test_time_fading.py --index "${SLURM_ARRAY_TASK_ID:?}"
