#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-benchmark
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-4%5
#SBATCH --output=results/phrase-recovery-16k/logs/benchmark-%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/phrase-recovery-16k/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PWD/.pixi/envs/default/bin/python" - <<'PY'
import os
import sys
sys.path.insert(0, "scripts")
from run_nine_loss_recovery import signature
assert signature()[0] == os.environ["SCIENTIFIC_SIGNATURE"]
PY
cardinalities=(1 2 4 6 8)
cardinality=${cardinalities[${SLURM_ARRAY_TASK_ID:?}]}
exec "$PWD/.pixi/envs/default/bin/python" scripts/benchmark_recovery_batch_scaling.py \
  --cardinality "$cardinality" --batches 150 \
  --output "results/phrase-recovery-16k/batch-scaling-c${cardinality}.json"
