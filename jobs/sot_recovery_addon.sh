#!/usr/bin/env bash
#SBATCH --job-name=sot-recovery
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=results/phrase-recovery-16k-sot/logs/%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/phrase-recovery-16k-sot/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mode=${MODE:-compute}
if [[ "$mode" == qualify ]]; then
  exec "$PWD/.pixi/envs/default/bin/python" scripts/run_sot_recovery_addon.py qualify \
    --device cuda
fi
"$PWD/.pixi/envs/default/bin/python" - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from run_nine_loss_recovery import signature
from run_sot_recovery_addon import plan_hash

qualification = json.loads(
    Path("results/phrase-recovery-16k-sot/qualification.json").read_text()
)
assert qualification["passed"]
assert qualification["scientific_signature"] == signature()[0]
assert qualification["execution_plan_sha256"] == plan_hash() == os.environ["EXECUTION_PLAN_SHA256"]
PY
exec "$PWD/.pixi/envs/default/bin/python" scripts/run_sot_recovery_addon.py compute \
  --device cuda --task "${SLURM_ARRAY_TASK_ID:?}"
