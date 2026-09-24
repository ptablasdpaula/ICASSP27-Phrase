"""Seven differentiable controls used by the paper's 7-D diagnostic."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._fractional import hermitian_rfft_projection, thiran_anchor
from .config import ExciterConfig, WaveguideConfig
from .waveguide import (
    _delay_rows,
    _df2_filter,
    _rational_add,
    _rational_multiply,
    _rational_scale,
    _RationalRows,
    _unwrapped_phase,
)

CONTROL_NAMES = (
    "log_f0",
    "onset",
    "log_amplitude",
    "log_duration",
    "pluck_position",
    "log_one_minus_gain",
    "loop_pole",
)


CONTROL_RANGES = {
    "f0_hz": (80.0, 320.0),
    "onset_seconds": (0.2, 1.8),
    "amplitude": (0.1, 1.0),
    "duration_seconds": (0.0025, 0.040),
    # Thiran-3 requires both physical rail sections to retain at least its
    # stable fractional-delay support at the highest registered pitch.
    "pluck_position": (0.10, 0.50),
    "loop_gain": (0.90, 0.9999),
    "loop_pole": (0.0, 0.75),
}


def _log_scale(unit: Tensor, lower: float, upper: float) -> Tensor:
    return torch.exp(math.log(lower) + unit * math.log(upper / lower))


def decode_controls(unit: Tensor) -> dict[str, Tensor]:
    if unit.ndim != 2 or unit.shape[1] != len(CONTROL_NAMES):
        raise ValueError("all-control coordinates must have shape [batch,7]")
    if not bool(((unit.detach() >= 0.0) & (unit.detach() <= 1.0)).all()):
        raise ValueError("all-control coordinates must lie in [0,1]")
    gain_lower, gain_upper = CONTROL_RANGES["loop_gain"]
    one_minus_gain = _log_scale(unit[:, 5], 1.0 - gain_lower, 1.0 - gain_upper)
    return {
        "f0_hz": 80.0 * torch.pow(2.0, 2.0 * unit[:, 0]),
        "onset_seconds": 0.2 + 1.6 * unit[:, 1],
        "amplitude": _log_scale(unit[:, 2], *CONTROL_RANGES["amplitude"]),
        "duration_seconds": _log_scale(unit[:, 3], *CONTROL_RANGES["duration_seconds"]),
        "pluck_position": 0.10 + 0.40 * unit[:, 4],
        "loop_gain": 1.0 - one_minus_gain,
        "loop_pole": 0.75 * unit[:, 6],
    }


def encode_defaults(*, device: torch.device | str) -> Tensor:
    exciter, waveguide = ExciterConfig(), WaveguideConfig()

    def log_unit(value: float, lower: float, upper: float) -> float:
        return math.log(value / lower) / math.log(upper / lower)

    gain_lower, gain_upper = CONTROL_RANGES["loop_gain"]
    values = [
        math.log2(160.0 / 80.0) / 2.0,
        (1.0 - 0.2) / 1.6,
        log_unit(exciter.amplitude, *CONTROL_RANGES["amplitude"]),
        log_unit(exciter.duration_seconds, *CONTROL_RANGES["duration_seconds"]),
        (waveguide.pluck_position - 0.10) / 0.40,
        log_unit(1.0 - waveguide.loop_gain, 1.0 - gain_lower, 1.0 - gain_upper),
        waveguide.loop_pole / 0.75,
    ]
    return torch.tensor(values, dtype=torch.float64, device=device)[None]


def dynamic_excitation(controls: dict[str, Tensor]) -> Tensor:
    config = ExciterConfig()
    reference = controls["f0_hz"]
    sample = torch.arange(config.sample_count, dtype=torch.float64, device=reference.device)[None]
    width = controls["duration_seconds"][:, None] * float(config.sample_rate)
    support = (sample < width.detach()).to(reference.dtype)
    prototype = (
        controls["amplitude"][:, None]
        * 0.5
        * (1.0 - torch.cos(torch.pi * sample / width))
        * support
    )
    spectrum = torch.fft.rfft(prototype, n=config.fourier_fft_length, dim=-1)
    omega = (
        2.0
        * torch.pi
        * torch.arange(
            config.fourier_fft_length // 2 + 1,
            dtype=reference.dtype,
            device=reference.device,
        )
        / float(config.fourier_fft_length)
    )
    phase = torch.exp(
        torch.complex(
            torch.zeros_like(controls["onset_seconds"][:, None] * omega[None]),
            -controls["onset_seconds"][:, None] * float(config.sample_rate) * omega[None],
        )
    )
    shifted = hermitian_rfft_projection(spectrum * phase, config.fourier_fft_length)
    return torch.fft.irfft(shifted, n=config.fourier_fft_length, dim=-1)[:, : config.sample_count]


def dynamic_transfer_rows(
    controls: dict[str, Tensor], config: WaveguideConfig
) -> tuple[Tensor, Tensor]:
    """Tensor-valued counterpart of the paper's fixed-config transfer builder."""
    frequency = controls["f0_hz"].reshape(-1)
    pole = controls["loop_pole"].reshape(-1)
    gain = controls["loop_gain"].reshape(-1)
    beta = controls["pluck_position"].reshape(-1)
    omega = 2.0 * math.pi * frequency / float(config.sample_rate)
    loop_phase = -torch.atan2(pole * torch.sin(omega), 1.0 - pole * torch.cos(omega))
    rail = 0.5 * (float(config.sample_rate) / frequency + loop_phase / omega)
    initial_long = (1.0 - beta) * rail
    initial_short = beta * rail
    long_order, short_order = config.segment_orders or (0, 0)
    if config.interpolation != "thiran" or (long_order, short_order) != (3, 3):
        raise ValueError("the all-control diagnostic requires the paper's Thiran-3 path")
    long_whole = thiran_anchor(initial_long, long_order)
    short_whole = thiran_anchor(initial_short, short_order)
    for _ in range(config.phase_correction_iterations):
        long_phase = _unwrapped_phase(
            (1.0 - beta) * rail,
            config.interpolation,
            long_order,
            long_whole,
            omega,
        )
        short_phase = _unwrapped_phase(
            beta * rail,
            config.interpolation,
            short_order,
            short_whole,
            omega,
        )
        residual = 2.0 * (long_phase + short_phase) + loop_phase + 2.0 * math.pi
        rail = rail + residual / (2.0 * omega)
    long = _delay_rows((1.0 - beta) * rail, config.interpolation, long_order, long_whole)
    short = _delay_rows(beta * rail, config.interpolation, short_order, short_whole)
    loop = _RationalRows(
        (gain * (1.0 - pole))[:, None],
        torch.stack((torch.ones_like(pole), -pole), dim=1),
    )
    rail_filter = _rational_multiply(long, short)
    feedback = _rational_multiply(loop, _rational_multiply(rail_filter, rail_filter))
    identity = _RationalRows(
        torch.ones_like(frequency[:, None]), torch.ones_like(frequency[:, None])
    )
    characteristic = _rational_add(identity, feedback, right_scale=-1.0)
    closed_loop = _RationalRows(characteristic.denominator, characteristic.numerator)
    nut_comb = _rational_add(identity, _rational_multiply(long, long), right_scale=-1.0)
    source_to_bridge = _rational_scale(_rational_multiply(short, nut_comb), 0.5)
    transfer = _rational_multiply(source_to_bridge, closed_loop)
    leading = transfer.denominator[:, :1]
    if bool((leading.detach().abs() < 1e-14).any()):
        raise FloatingPointError("dynamic waveguide transfer is singular")
    return transfer.numerator / leading, transfer.denominator[:, 1:] / leading


def render_all_controls(unit: Tensor) -> Tensor:
    controls = decode_controls(unit)
    source = dynamic_excitation(controls)
    numerator, denominator = dynamic_transfer_rows(controls, WaveguideConfig())
    result = _df2_filter(source, numerator, denominator)
    if not bool(torch.isfinite(result.detach()).all()):
        raise FloatingPointError("all-control renderer produced non-finite audio")
    return result
