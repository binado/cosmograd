"""Perturbation protocol and implementations.

Import emulator modules explicitly; optional dependencies load on first use.
"""

from .eisenstein_hu import EisensteinHu
from .growth import linear_growth
from .protocol import Perturbations

__all__ = ["EisensteinHu", "Perturbations", "linear_growth"]
