#!/usr/bin/env bash
#SBATCH --job-name=cel-grad-fixed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH --output=results/gradient-assessment-16k/logs/%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/gradient-assessment-16k/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mode=${MODE:-compute}
if [[ "$mode" == qualify ]]; then
  "$PWD/.pixi/envs/default/bin/python" - <<'PY'
import os
import sys
sys.path.insert(0, "scripts")
from fixed_gradient_assessment import signature
assert signature()[0] == os.environ["SCIENTIFIC_SIGNATURE"]
PY
  exec "$PWD/.pixi/envs/default/bin/python" scripts/run_fixed_gradient_assessment.py qualify \
    --device cuda --batch "${BATCH_SIZE:-16}"
fi
if [[ "$mode" == compute ]]; then
  "$PWD/.pixi/envs/default/bin/python" - <<'PY'
import json
import os
from pathlib import Path

qualification = json.loads(
    Path("results/gradient-assessment-16k/qualification.json").read_text()
)
assert qualification["passed"]
assert qualification["signature"] == os.environ["SCIENTIFIC_SIGNATURE"]
PY
  task=${SLURM_ARRAY_TASK_ID:?}
  cardinalities=(1 2 4 6 8)
  cardinality=${cardinalities[$((task / 32))]}
  target_index=$((task % 32))
  exec "$PWD/.pixi/envs/default/bin/python" scripts/run_fixed_gradient_assessment.py compute \
    --device cuda --batch "${BATCH_SIZE:-16}" --cardinality "$cardinality" \
    --target-index "$target_index"
fi
echo "Unknown MODE=$mode" >&2
exit 2
