"""Reproduction tools for cumulative energy losses and musical phrase recovery."""

from .data import Target, load_target
from .losses import LOSS_LABELS, PaperObjectives
from .optimization import FitResult, FitSnapshot, Schedule, fit
from .runtime import configure_reproducibility, require_df2_backend
from .synth import EventPhrase, Exciter, ExciterConfig, PhraseSynth, Waveguide, WaveguideConfig

__all__ = [
    "EventPhrase",
    "Exciter",
    "ExciterConfig",
    "PhraseSynth",
    "Waveguide",
    "WaveguideConfig",
    "Target",
    "load_target",
    "LOSS_LABELS",
    "PaperObjectives",
    "Schedule",
    "FitResult",
    "FitSnapshot",
    "fit",
    "configure_reproducibility",
    "require_df2_backend",
]
