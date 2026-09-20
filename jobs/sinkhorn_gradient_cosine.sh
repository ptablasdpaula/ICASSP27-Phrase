#!/usr/bin/env bash
#SBATCH --job-name=tf-sink-grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=results/gradient-assessment-sinkhorn/logs/%A_%a.log

set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
mkdir -p results/gradient-assessment-sinkhorn/logs
module load cuda/12.6.2-gcc-12.2.0
# The pinned CUDA DF2 extensions are built by ``pixi run install-backends`` in
# the default environment; the Sinkhorn environment adds GeomLoss/PyKeOps.
export PYTHONPATH="$PWD/.pixi/envs/default/lib/python3.12/site-packages:$PWD/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export KEOPS_CACHE_FOLDER="/data/home/acw794/.cache/keops-icaspp27"
export PYTHONHOME="$PWD/.pixi/envs/sinkhorn" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mode=${MODE:-compute}
if [[ "$mode" == qualify ]]; then
  exec "$PWD/.pixi/envs/sinkhorn/bin/python" scripts/assess_sinkhorn_gradient.py qualify \
    --device cuda --batch "${BATCH_SIZE:-2}"
fi
if [[ "$mode" == compute ]]; then
  task=${SLURM_ARRAY_TASK_ID:?}
  if (( task < 160 )); then
    cardinalities=(1 2 4 6 8)
    cardinality=${cardinalities[$((task / 32))]}
    target=$((task % 32))
    case "$cardinality" in
      1) default_batch=32 ;;
      2) default_batch=24 ;;
      4) default_batch=16 ;;
      6) default_batch=12 ;;
      8) default_batch=8 ;;
    esac
    exec "$PWD/.pixi/envs/sinkhorn/bin/python" scripts/assess_sinkhorn_gradient.py compute \
      --device cuda --batch "${BATCH_SIZE:-$default_batch}" --kind standard \
      --cardinality "$cardinality" --target-index "$target"
  elif (( task < 192 )); then
    target=$((task - 160))
    exec "$PWD/.pixi/envs/sinkhorn/bin/python" scripts/assess_sinkhorn_gradient.py compute \
      --device cuda --batch "${BATCH_SIZE:-24}" --kind persistent_target \
      --target-index "$target"
  else
    target=$((task - 192))
    exec "$PWD/.pixi/envs/sinkhorn/bin/python" scripts/assess_sinkhorn_gradient.py compute \
      --device cuda --batch "${CONTROL_BATCH_SIZE:-8}" --kind all_controls \
      --target-index "$target"
  fi
fi
echo "Unknown MODE=$mode" >&2
exit 2
