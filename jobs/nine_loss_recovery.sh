#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-recovery
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH --output=results/nine-loss-recovery/logs/%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/nine-loss-recovery/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PWD/.pixi/envs/default/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from run_nine_loss_recovery import signature
q = json.loads(Path("results/nine-loss-recovery/qualification-cuda.json").read_text())
actual = signature()[0]
assert q["passed"] and q["signature"] == actual
assert actual == os.environ["SCIENTIFIC_SIGNATURE"]
PY
exec "$PWD/.pixi/envs/default/bin/python" scripts/run_nine_loss_recovery.py --shard "${SLURM_ARRAY_TASK_ID:?}"
