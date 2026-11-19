"""Perturbation protocol and implementations.

Import emulator modules explicitly; optional dependencies load on first use.
"""

from .eisenstein_hu import EisensteinHu, EisensteinHuParameters
from .growth import linear_growth, matter_domination_scale_factor
from .protocol import KRange, Perturbations

__all__ = [
    "EisensteinHu",
    "EisensteinHuParameters",
    "KRange",
    "Perturbations",
    "linear_growth",
    "matter_domination_scale_factor",
]
