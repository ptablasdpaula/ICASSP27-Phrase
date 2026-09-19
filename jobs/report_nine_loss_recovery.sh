#!/usr/bin/env bash
#SBATCH --job-name=cel-nine-report
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH --output=results/phrase-recovery-16k/report-%j.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PWD/.pixi/envs/default/bin/python" scripts/report_nine_loss_recovery.py --device cuda
