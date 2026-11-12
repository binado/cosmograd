"""Differentiable cosmology in JAX behind a small protocol contract.

Consumers depend on the :class:`Background` and :class:`Perturbations`
protocols; :class:`FlatLCDM` is the native background implementation.
Emulator backends live in :mod:`cosmograd.perturbations` and are never imported
here.
"""

from importlib.metadata import version

from .background import Background, FlatLCDM
from .perturbations import Perturbations

__version__ = version("cosmograd")

__all__ = ["Background", "FlatLCDM", "Perturbations", "__version__"]
