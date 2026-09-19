#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-packed-qualify
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=results/phrase-recovery-16k/logs/qualify-%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/phrase-recovery-16k/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PWD/.pixi/envs/default/bin/python" - <<'PY'
import os
import sys
sys.path.insert(0, "scripts")
from run_nine_loss_recovery import signature
assert signature()[0] == os.environ["SCIENTIFIC_SIGNATURE"]
PY
exec "$PWD/.pixi/envs/default/bin/python" scripts/qualify_packed_nine_loss_recovery.py --device cuda
