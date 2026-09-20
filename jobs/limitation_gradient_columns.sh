#!/usr/bin/env bash
#SBATCH --job-name=cel-limit-grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH --output=results/gradient-assessment-limitations/logs/%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/gradient-assessment-limitations/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mode=${MODE:-compute}
if [[ "$mode" == qualify ]]; then
  exec "$PWD/.pixi/envs/default/bin/python" scripts/assess_limitation_columns.py qualify \
    --device cuda
fi
if [[ "$mode" == compute ]]; then
  task=${SLURM_ARRAY_TASK_ID:?}
  if (( task < 32 )); then
    condition=persistent_target
    target=$task
    batch=${PERSISTENT_BATCH_SIZE:-16}
  else
    condition=all_controls
    target=$((task - 32))
    batch=${ALL_CONTROL_BATCH_SIZE:-8}
  fi
  exec "$PWD/.pixi/envs/default/bin/python" scripts/assess_limitation_columns.py compute \
    --device cuda --batch "$batch" --condition "$condition" --target-index "$target"
fi
echo "Unknown MODE=$mode" >&2
exit 2
