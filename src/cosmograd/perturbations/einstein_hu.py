"""Eisenstein-Hu linear matter power spectrum in pure JAX.

Not implemented yet. This module is a placeholder for the native
:class:`~cosmograd.Perturbations` implementation installed with the core.
"""

from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..background import Background
from .protocol import Perturbations

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

    def growth_function(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def growth_rate(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
        *,
        nonlinear: bool = False,
    ) -> Float[Array, "*shape"]:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError

    def primordial_power_spectrum(
        self,
        k: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError
