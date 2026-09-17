#!/usr/bin/env bash
set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PWD/.pixi/envs/default/bin/python" scripts/extend_gradient_cardinality.py --batch 64
