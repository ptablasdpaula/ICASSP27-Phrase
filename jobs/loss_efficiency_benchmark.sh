#!/usr/bin/env bash
#SBATCH --job-name=loss-efficiency
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --constraint=ampere
#SBATCH --output=results/loss-efficiency/benchmark-%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/loss-efficiency
export PYTHONHOME="$PWD/.pixi/envs/default"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
losses=(single_stft smooth_mss linear_jtfot cel)
task=${SLURM_ARRAY_TASK_ID:-0}
loss=${losses[$((task % ${#losses[@]}))]}
exec "$PWD/.pixi/envs/default/bin/python" scripts/benchmark_loss_efficiency.py \
  --cardinalities 1 \
  --losses "$loss" \
  --target-count 150 --warmup 5 --measured 20 --repeats 1 \
  --output "results/loss-efficiency/benchmark-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.json"
