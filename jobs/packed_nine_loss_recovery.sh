#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-packed
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH --output=results/nine-loss-recovery-packed/logs/%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/nine-loss-recovery-packed/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PWD/.pixi/envs/default/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from run_nine_loss_recovery import signature
from run_packed_nine_loss_recovery import plan_hash
q = json.loads(Path("results/nine-loss-recovery-packed/qualification-cuda.json").read_text())
assert q["passed"] and q["scientific_signature"] == signature()[0]
assert q["execution_plan_sha256"] == plan_hash() == os.environ["EXECUTION_PLAN_SHA256"]
PY
exec "$PWD/.pixi/envs/default/bin/python" scripts/run_packed_nine_loss_recovery.py --task "${SLURM_ARRAY_TASK_ID:?}"
