#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-benchmark
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/nine-loss-recovery/logs/benchmark-%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/nine-loss-recovery/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PWD/.pixi/envs/default/bin/python" scripts/benchmark_nine_loss_recovery.py
