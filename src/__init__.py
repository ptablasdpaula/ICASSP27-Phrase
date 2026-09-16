"""Public API for the ICASSP 2027 phrase-recovery renderer."""

from .config import (
    CARDINALITIES,
    EventPhrase,
    ExciterConfig,
    OptimizerConfig,
    WaveguideConfig,
    initial_candidate,
)
from .exciter import Exciter
from .losses import (
    CEL_DIRECTIONS,
    CEL_NAMES,
    LOSS_LABELS,
    PAPER_LOSSES,
    CumulativeEnergyDistance,
    build_loss,
)
from .optimization import FitResult, FitSnapshot, fit
from .runtime import configure_reproducibility, require_df2_backend
from .synth import PhraseSynth
from .targets import Target, load_target
from .waveguide import Waveguide, WaveguideTransfer

__all__ = [
    "CARDINALITIES",
    "CEL_DIRECTIONS",
    "CEL_NAMES",
    "CumulativeEnergyDistance",
    "EventPhrase",
    "Exciter",
    "ExciterConfig",
    "FitResult",
    "FitSnapshot",
    "LOSS_LABELS",
    "OptimizerConfig",
    "PAPER_LOSSES",
    "PhraseSynth",
    "Target",
    "Waveguide",
    "WaveguideConfig",
    "WaveguideTransfer",
    "build_loss",
    "configure_reproducibility",
    "fit",
    "initial_candidate",
    "load_target",
    "require_df2_backend",
]
