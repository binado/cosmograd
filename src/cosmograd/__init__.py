"""Differentiable cosmology in JAX behind a small protocol contract.

Consumers depend on the :class:`Background` and :class:`Perturbations`
protocols; :class:`LCDM`, :class:`W0WaCDM`, :class:`FlatLCDM` and
:class:`FlatW0WaCDM` are the native backgrounds. Standard cosmologies such as ``Planck18`` live in
:mod:`cosmograd.cosmologies`.
Emulator backends live in :mod:`cosmograd.perturbations` and are never imported
here.
"""

from importlib.metadata import version

from . import cosmologies
from .background import LCDM, Background, FlatLCDM, FlatW0WaCDM, W0WaCDM
from .perturbations import EisensteinHu, Perturbations

__version__ = version("cosmograd")

__all__ = [
    "LCDM",
    "Background",
    "EisensteinHu",
    "FlatLCDM",
    "FlatW0WaCDM",
    "Perturbations",
    "W0WaCDM",
    "cosmologies",
    "__version__",
]
