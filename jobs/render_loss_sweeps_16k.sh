#!/usr/bin/env bash
#SBATCH --job-name=cel-sweeps-16k
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/loss-sweeps-16k-%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PWD/.pixi/envs/default/bin/python" - <<'PY'
import os
import sys
sys.path.insert(0, "scripts")
from render_loss_sweeps import scientific_signature
assert scientific_signature()[0] == os.environ["SCIENTIFIC_SIGNATURE"]
PY
exec "$PWD/.pixi/envs/default/bin/python" scripts/render_loss_sweeps.py \
  --device cuda --batch-size 32
