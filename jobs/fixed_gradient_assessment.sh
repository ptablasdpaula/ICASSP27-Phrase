#!/usr/bin/env bash
#SBATCH --job-name=cel-grad-fixed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH --output=results/gradient-assessment-fixed-lhs/logs/%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/gradient-assessment-fixed-lhs/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
actual_commit=$(git rev-parse HEAD)
if [[ "${SOURCE_COMMIT:?}" != "$actual_commit" ]]; then
  echo "Source commit changed: expected $SOURCE_COMMIT, found $actual_commit" >&2
  exit 1
fi
exec "$PWD/.pixi/envs/default/bin/python" scripts/run_fixed_gradient_assessment.py all \
  --device cuda --batch 64
