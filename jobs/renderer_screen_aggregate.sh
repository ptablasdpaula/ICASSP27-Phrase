#!/usr/bin/env bash
# Assemble browser JSON after all sixteen GPU cells have completed.
set -euo pipefail

repo=${ICASSP27_REPOSITORY:?}
source_commit=${ICASSP27_SOURCE_COMMIT:?}
output_root=${ICASSP27_RENDERER_SCREEN_ROOT:?}

test "${ICASSP27_JOB_CLASS:?}" = cpu || {
  echo "Renderer-screen assembly expects the CPU finalizer." >&2
  exit 66
}
test "$(git -C "$repo" rev-parse HEAD)" = "$source_commit" || {
  echo "Renderer-screen source commit changed after submission." >&2
  exit 65
}
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" || {
  echo "Renderer-screen assembly refuses a dirty source worktree." >&2
  exit 65
}

export PYTHONHASHSEED=2027
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH="$repo/.pixi/envs/default/bin:$PATH"
export LD_LIBRARY_PATH="$repo/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$repo/.pixi/envs/default/bin/python" scripts/build_renderer_screen.py \
  assemble \
  --input-root "$output_root" \
  --output-root "$output_root/web-data"
