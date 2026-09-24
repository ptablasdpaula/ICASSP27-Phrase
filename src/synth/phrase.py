"""Composition of the paper exciter and waveguide modules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from torch import Tensor, nn

from .config import EventPhrase, ExciterConfig, WaveguideConfig
from .exciter import Exciter
from .waveguide import Waveguide


class PhraseSynth(nn.Module):
    """Render known-cardinality phrases as ``Exciter -> Waveguide``.

    The default constructor is exactly the recovery renderer in the paper:
    FLAMO Fourier placement of the half-raised cosine, followed by the
    Thiran-3/Thiran-3 waveguide with a hard reset at every event boundary.
    PhilTorch dispatches the DF2 recurrence to TorchLPC on both CPU and CUDA.
    """

    def __init__(
        self,
        exciter: Exciter | None = None,
        waveguide: Waveguide | None = None,
        *,
        exciter_config: ExciterConfig | None = None,
        waveguide_config: WaveguideConfig | None = None,
    ) -> None:
        super().__init__()
        if exciter is not None and exciter_config is not None:
            raise ValueError("pass an Exciter or its config, not both")
        if waveguide is not None and waveguide_config is not None:
            raise ValueError("pass a Waveguide or its config, not both")
        self.exciter = exciter or Exciter(exciter_config)
        self.waveguide = waveguide or Waveguide(waveguide_config)
        if (
            self.exciter.config.sample_rate != self.waveguide.config.sample_rate
            or self.exciter.config.sample_count != self.waveguide.config.sample_count
        ):
            raise ValueError("Exciter and Waveguide sample grids must match")

    @property
    def sample_rate(self) -> int:
        return self.exciter.config.sample_rate

    @property
    def sample_count(self) -> int:
        return self.exciter.config.sample_count

    def source(self, onset_seconds: Tensor) -> Tensor:
        """Expose the summed excitation for diagnostics."""
        return self.exciter(onset_seconds)

    def render_batch(self, f0_hz: Tensor, onset_seconds: Tensor) -> Tensor:
        """Render controls shaped ``[batch,event]``."""
        if f0_hz.ndim != 2 or onset_seconds.shape != f0_hz.shape:
            raise ValueError("controls must have matching [batch,event] shapes")
        order = onset_seconds.detach().argsort(dim=1, stable=True)
        ordered_f0 = f0_hz.gather(1, order)
        ordered_onset = onset_seconds.gather(1, order)
        source = self.exciter(ordered_onset)
        return self.waveguide(source, ordered_f0, ordered_onset)

    def render(self, phrase: EventPhrase) -> Tensor:
        """Render one :class:`EventPhrase`."""
        return self.render_batch(phrase.f0_hz[None, :], phrase.onset_seconds[None, :])[0]

    def forward(self, f0_hz: Tensor, onset_seconds: Tensor) -> Tensor:
        return self.render_batch(f0_hz, onset_seconds)

    def provenance(self) -> dict[str, Any]:
        """Return a compact, JSON-safe description of the active path."""
        return {
            "schema": "icassp27-phrase-renderer-v1",
            "architecture": ["Exciter", "Waveguide", "PhraseSynth"],
            "exciter": asdict(self.exciter.config),
            "waveguide": asdict(self.waveguide.config),
            "dtype": "float64/complex128",
            "candidate_coordinates": (
                "independent elementwise bounded logits; no ordering, spacing, "
                "canonicalization, or Magic Clamp"
            ),
        }


__all__ = ["PhraseSynth"]
