#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-packed-qualify
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=results/nine-loss-recovery-packed/logs/qualify-%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/nine-loss-recovery-packed/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PWD/.pixi/envs/default/bin/python" scripts/qualify_packed_nine_loss_recovery.py --device cuda
