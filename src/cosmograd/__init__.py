"""Differentiable cosmology in JAX behind a small protocol contract.

Consumers depend on the :class:`Background` and :class:`Perturbations`
protocols; :class:`FlatLCDM` and :class:`FlatW0WaCDM` are the native
backgrounds.
Emulator backends live in :mod:`cosmograd.perturbations` and are never imported
here.
"""

from importlib.metadata import version

from .background import Background, FlatLCDM, FlatW0WaCDM
from .perturbations import EisensteinHu, Perturbations

__version__ = version("cosmograd")

__all__ = [
    "Background",
    "EisensteinHu",
    "FlatLCDM",
    "FlatW0WaCDM",
    "Perturbations",
    "__version__",
]
