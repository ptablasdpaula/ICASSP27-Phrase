"""Full Sinkhorn time--frequency OT from Fabiani et al. (Asilomar 2024)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    from geomloss import SamplesLoss
except ImportError as error:  # pragma: no cover - cluster dependency check
    raise ImportError(
        "the full Sinkhorn diagnostic requires geomloss==0.2.6 and pykeops==2.3"
    ) from error

from icassp27_phrase.config import SAMPLE_RATE
from icassp27_phrase.losses import TFW_HOP, TFW_HZ_PER_SECOND, TFW_N_FFT, _stft

SINKHORN_EPSILON = 1e-4
SINKHORN_BLUR = math.sqrt(SINKHORN_EPSILON)
SINKHORN_SCALING = 0.5
SINKHORN_TRUNCATE = 5.0
SINKHORN_CLUSTER_SCALE = 0.03


def _normalise_mass(value: Tensor) -> Tensor:
    mass = value.sum(dim=(-2, -1), keepdim=True)
    if bool((mass.detach() <= 0.0).any()):
        raise ValueError("Sinkhorn spectrograms must have positive total magnitude")
    return value / mass


def _support(
    frequency_bins: int,
    time_bins: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, float]:
    """Return coordinates whose GeomLoss cost equals the paper's unit-max cost.

    GeomLoss uses ``0.5 * ||x-y||^2`` for ``p=2``. Scaling both published
    coordinates by ``sqrt(2 / max(C))`` therefore gives ``C / max(C)`` while
    retaining the published trade-off of one second against 1000 Hz.
    """
    frequency = torch.arange(frequency_bins, dtype=dtype, device=device) * (
        SAMPLE_RATE / TFW_N_FFT
    )
    time = (
        torch.arange(time_bins, dtype=dtype, device=device)
        * (TFW_HOP / SAMPLE_RATE)
        * TFW_HZ_PER_SECOND
    )
    frequency_grid, time_grid = torch.meshgrid(frequency, time, indexing="ij")
    maximum_cost = float((frequency[-1].square() + time[-1].square()).cpu())
    points = torch.stack((frequency_grid, time_grid), dim=-1).reshape(-1, 2)
    return points * math.sqrt(2.0 / maximum_cost), maximum_cost


class SinkhornTFWObjective:
    """Target-bound full entropic joint time--frequency transport objective.

    The Asilomar paper defines the entropy-regularised cost rather than the
    debiased Sinkhorn divergence, hence ``debias=False``. GeomLoss' multiscale
    backend supplies log-domain stabilisation and epsilon scaling at the
    published final regularisation without constructing the dense plan.
    """

    def __init__(self, target: Tensor) -> None:
        if target.ndim == 1:
            target = target[None]
        if target.ndim != 2 or len(target) != 1 or not target.is_floating_point():
            raise ValueError("target must contain one floating-point audio row")
        self.target = target.detach()
        self.window = torch.hann_window(
            TFW_N_FFT,
            periodic=True,
            dtype=target.dtype,
            device=target.device,
        )
        with torch.no_grad():
            magnitude = _stft(
                self.target,
                n_fft=TFW_N_FFT,
                hop=TFW_HOP,
                window=self.window,
                center=True,
            ).abs()
            self.target_mass = _normalise_mass(magnitude)[0].flatten()
        self.points, self.maximum_cost = _support(
            magnitude.shape[-2],
            magnitude.shape[-1],
            dtype=target.dtype,
            device=target.device,
        )
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=SINKHORN_BLUR,
            scaling=SINKHORN_SCALING,
            truncate=SINKHORN_TRUNCATE,
            diameter=math.sqrt(2.0),
            cluster_scale=SINKHORN_CLUSTER_SCALE,
            debias=False,
            potentials=False,
            backend="multiscale",
        )

    def __call__(self, candidate: Tensor) -> Tensor:
        if candidate.ndim == 1:
            candidate = candidate[None]
        magnitude = _stft(
            candidate,
            n_fft=TFW_N_FFT,
            hop=TFW_HOP,
            window=self.window,
            center=True,
        ).abs()
        masses = _normalise_mass(magnitude).flatten(1)
        # The multiscale backend intentionally rejects a batch dimension. Each
        # scalar stays in the common autograd graph, so a summed backward pass
        # still yields all independent candidate gradients.
        return torch.stack(
            [
                self.sinkhorn(source, self.points, self.target_mass, self.points)
                for source in masses
            ]
        )

    def provenance(self) -> dict[str, object]:
        return {
            "representation": "globally-normalised Hann STFT magnitude",
            "n_fft": TFW_N_FFT,
            "hop_length": TFW_HOP,
            "center": True,
            "time_scale_hz_per_second": TFW_HZ_PER_SECOND,
            "squared_ground_cost": True,
            "cost_normalisation": "divide by maximum squared joint-grid distance",
            "maximum_unnormalised_cost_hz2": self.maximum_cost,
            "epsilon": SINKHORN_EPSILON,
            "implementation": "GeomLoss 0.2.6 multiscale PyKeOps backend",
            "debias": False,
            "epsilon_scaling_ratio": SINKHORN_SCALING,
            "kernel_truncation_sigma": SINKHORN_TRUNCATE,
            "cluster_scale": SINKHORN_CLUSTER_SCALE,
        }


__all__ = [
    "SINKHORN_BLUR",
    "SINKHORN_CLUSTER_SCALE",
    "SINKHORN_EPSILON",
    "SINKHORN_SCALING",
    "SINKHORN_TRUNCATE",
    "SinkhornTFWObjective",
]
