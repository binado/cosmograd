"""Eisenstein-Hu linear matter power spectrum in pure JAX.

Not implemented yet. This module is a placeholder for the native
:class:`~cosmograd.Perturbations` implementation installed with the core.
"""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike

from .protocols import Background, Perturbations

__all__ = ["EisensteinHu"]


class EisensteinHu(Perturbations):
    """Eisenstein & Hu (1998) transfer function with a power-law primordial spectrum.

    Placeholder: every method raises :class:`NotImplementedError`.
    """

    @property
    def background(self) -> Background:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def growth_function(self, z: ArrayLike, *, k: ArrayLike | None = None) -> Array:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def growth_rate(self, z: ArrayLike, *, k: ArrayLike | None = None) -> Array:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def matter_power_spectrum(
        self, k: ArrayLike, z: ArrayLike, *, nonlinear: bool = False
    ) -> Array:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def primordial_power_spectrum(self, k: ArrayLike) -> Array:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError
