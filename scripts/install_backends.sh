#!/usr/bin/env bash
# Build the exact PhilTorch/TorchLPC route used for the paper's DF2 recurrence.
set -euo pipefail

torchlpc_commit=1bfde4a457f87b1dd0fc22a6548206be3a26647c
philtorch_commit=710946142b6149b486a37f4a3be87ddbf9e2cda3
backend_device="${ICASSP27_BACKEND_DEVICE:-auto}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
case "$backend_device" in
  auto)
    backend_device=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
    ;;
  cpu|cuda) ;;
  *)
    echo "ICASSP27_BACKEND_DEVICE must be auto, cpu, or cuda" >&2
    exit 64
    ;;
esac
export ICASSP27_BACKEND_DEVICE_ACTUAL="$backend_device"
if [ "$backend_device" = "cuda" ]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required; load a CUDA 12.6 toolkit before building CUDA" >&2
    exit 69
  fi
else
  export CUDA_VISIBLE_DEVICES=""
fi
backend_build=$(mktemp -d "${TMPDIR:-/tmp}/icassp27-phrase-backends.XXXXXX")
cleanup() {
  case "$backend_build" in
    "${TMPDIR:-/tmp}"/icassp27-phrase-backends.*) rm -rf -- "$backend_build" ;;
    *) echo "Refusing to clean an unexpected build path" >&2; return 68 ;;
  esac
}
trap cleanup EXIT

python - <<'PY'
import os
import torch

device = os.environ.get("ICASSP27_BACKEND_DEVICE_ACTUAL", "cpu")
if torch.__version__.partition("+")[0] != "2.7.1":
    raise RuntimeError(
        f"expected torch 2.7.1, got {torch.__version__}"
    )
if device == "cuda" and (
    torch.__version__ != "2.7.1+cu126"
    or torch.version.cuda != "12.6"
    or not torch.cuda.is_available()
):
    raise RuntimeError(
        "CUDA builds require visible CUDA with torch 2.7.1+cu126/CUDA 12.6; "
        f"got {torch.__version__}/{torch.version.cuda}"
    )
PY

git clone --quiet https://github.com/DiffAPF/torchlpc.git "$backend_build/torchlpc"
git -C "$backend_build/torchlpc" checkout --quiet "$torchlpc_commit"
git clone --quiet https://github.com/yoyolicoris/philtorch.git "$backend_build/philtorch"
git -C "$backend_build/philtorch" checkout --quiet "$philtorch_commit"

python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$backend_build/wheels" "$backend_build/torchlpc"
# This project uses PhilTorch as the public LPV frontend and TorchLPC as the
# CPU/CUDA recurrence. Building the independent PhilTorch extension with CUDA
# hidden avoids a removed CUDA/Thrust API in the registered toolchain; its
# Python LPV frontend still dispatches the recurrence through TorchLPC.
CUDA_VISIBLE_DEVICES= python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$backend_build/wheels" "$backend_build/philtorch"
python -m pip install --no-deps --force-reinstall "$backend_build"/wheels/torchlpc-*.whl
python -m pip install --no-deps --force-reinstall "$backend_build"/wheels/philtorch-*.whl

python - <<'PY'
import os
import torch

from icassp27_phrase.runtime import require_df2_backend

device = torch.device(os.environ.get("ICASSP27_BACKEND_DEVICE_ACTUAL", "cpu"))
require_df2_backend(device)
summary = {"device": device.type, "torch": torch.__version__}
if device.type == "cuda":
    summary.update({"cuda": torch.version.cuda,
                    "architectures": torch._C._cuda_getArchFlags().split()})
print(summary)
PY
