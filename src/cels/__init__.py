"""PyTorch Cumulative Energy Losses."""

from .loss import (
    ALL_DIRECTIONS,
    DIAGONAL_DIRECTIONS,
    ORTHOGONAL_DIRECTIONS,
    BoundCumulativeEnergyLoss,
    CumulativeEnergyLoss,
    Direction,
)
from .stft import (
    BoundSTFTCumulativeEnergyLoss,
    STFTCumulativeEnergyLoss,
    STFTPower,
    paper_loss,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_DIRECTIONS",
    "DIAGONAL_DIRECTIONS",
    "ORTHOGONAL_DIRECTIONS",
    "BoundCumulativeEnergyLoss",
    "BoundSTFTCumulativeEnergyLoss",
    "CumulativeEnergyLoss",
    "Direction",
    "STFTCumulativeEnergyLoss",
    "STFTPower",
    "paper_loss",
]
