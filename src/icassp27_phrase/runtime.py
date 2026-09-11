"""Small fail-closed runtime contract for the paper renderer."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .config import MASTER_SEED

CUDA_WORKSPACE_CONFIG = ":4096:8"
TORCH_VERSION = "2.7.1+cu126"
TORCH_BASE_VERSION = "2.7.1"
CUDA_VERSION = "12.6"
TORCHLPC_COMMIT = "1bfde4a457f87b1dd0fc22a6548206be3a26647c"
PHILTORCH_COMMIT = "710946142b6149b486a37f4a3be87ddbf9e2cda3"


def configure_reproducibility(seed: int = MASTER_SEED) -> None:
    """Configure the deterministic controls registered by the study."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUDA_WORKSPACE_CONFIG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def require_df2_backend(device: torch.device | str) -> None:
    """Require PhilTorch to dispatch DF2 through a compiled TorchLPC kernel."""
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise RuntimeError("the DF2 realization supports only CPU or CUDA")
    if torch.__version__.partition("+")[0] != TORCH_BASE_VERSION:
        raise RuntimeError(
            f"expected PyTorch {TORCH_BASE_VERSION}, found {torch.__version__}"
        )
    if target.type == "cuda" and (
        not torch.cuda.is_available()
        or torch.__version__ != TORCH_VERSION
        or torch.version.cuda != CUDA_VERSION
    ):
        raise RuntimeError(
            f"expected PyTorch {TORCH_VERSION}/CUDA {CUDA_VERSION}, "
            f"found {torch.__version__}/{torch.version.cuda}"
        )
    try:
        import torchlpc
        from philtorch.lpv import filtering as philtorch_filtering
        from torchlpc import core as torchlpc_core
    except Exception as error:  # pragma: no cover - installation-specific
        raise RuntimeError("install the pinned PhilTorch and TorchLPC backends") from error
    if not bool(getattr(torchlpc, "EXTENSION_LOADED", False)) or not bool(
        getattr(torchlpc_core, "EXTENSION_LOADED", False)
    ):
        raise RuntimeError("TorchLPC selected a fallback instead of its compiled extension")
    if philtorch_filtering.sample_wise_lpc is not torchlpc.sample_wise_lpc:
        raise RuntimeError("PhilTorch is not routed to TorchLPC")
    try:
        has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(
            "torchlpc::lpc", target.type.upper()
        )
    except Exception as error:  # pragma: no cover - installation-specific
        raise RuntimeError("TorchLPC did not register its LPC operator") from error
    if not has_kernel:
        raise RuntimeError(
            f"TorchLPC did not register the required {target.type.upper()} recurrence"
        )


def require_accelerated_backend(device: torch.device | str) -> None:
    """Require the paper-qualified PhilTorch/TorchLPC CUDA route."""
    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError("the accelerated backend requires a CUDA device")
    require_df2_backend(target)


__all__ = [
    "CUDA_VERSION",
    "CUDA_WORKSPACE_CONFIG",
    "PHILTORCH_COMMIT",
    "TORCH_BASE_VERSION",
    "TORCHLPC_COMMIT",
    "TORCH_VERSION",
    "configure_reproducibility",
    "require_accelerated_backend",
    "require_df2_backend",
]
