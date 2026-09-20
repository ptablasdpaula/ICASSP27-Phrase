#!/usr/bin/env bash
#SBATCH --job-name=loss-efficiency
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=results/loss-efficiency/benchmark-%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/loss-efficiency
export PYTHONHOME="$PWD/.pixi/envs/default"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PWD/.pixi/envs/default/bin/python" scripts/benchmark_loss_efficiency.py \
  --cardinalities 1 \
  --output "results/loss-efficiency/benchmark-${SLURM_JOB_ID}.json"
