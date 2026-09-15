#!/usr/bin/env bash
# One CUDA-only renderer cell; resources and identity come from the launcher.
set -euo pipefail

repo=${ICASSP27_REPOSITORY:?}
source_commit=${ICASSP27_SOURCE_COMMIT:?}
output_root=${ICASSP27_RENDERER_SCREEN_ROOT:?}

test "${ICASSP27_JOB_CLASS:?}" = gpu || {
  echo "Renderer-screen waveform evaluation requires a GPU job." >&2
  exit 66
}
test "$(git -C "$repo" rev-parse HEAD)" = "$source_commit" || {
  echo "Renderer-screen source commit changed after submission." >&2
  exit 65
}
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" || {
  echo "Renderer-screen evaluation refuses a dirty source worktree." >&2
  exit 65
}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=2027
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="$repo/.pixi/envs/default/bin:$PATH"
export LD_LIBRARY_PATH="$repo/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$repo/.pixi/envs/default/bin/python" scripts/build_renderer_screen.py \
  compute-cell \
  --index "${SLURM_ARRAY_TASK_ID:?}" \
  --output-root "$output_root" \
  --device cuda \
  --surface-chunk-size 16 \
  --gradient-chunk-size 4
