#!/usr/bin/env bash
#SBATCH --job-name=cel-log-dec-dir
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/log-decay-direction-gradient/logs/%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/log-decay-direction-gradient/logs
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PWD/.pixi/envs/default/bin/python" scripts/assess_log_decay_directions.py all --batch 64
