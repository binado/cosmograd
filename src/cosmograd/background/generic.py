"""Background built from a Hubble rate and parameters the caller owns."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from . import functional
from .protocol import Background

__all__ = ["GenericBackground"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["params"],
    meta_fields=["hubble", "omega_m", "omega_k"],
)
@dataclass(frozen=True)
class GenericBackground[P](Background):
    """Background over any pytree of parameters and a Hubble rate.

    The instance is a pytree whose leaves are those of ``params``, so
    gradients and batches come back in the caller's own layout, e.g.
    ``jax.grad(lambda b: b.luminosity_distance(z))(background).params``.

    The three callables are static. JAX compares them by identity, so define
    them once, at module level; a ``lambda`` written inline in the constructor
    call makes every instance a new input to :func:`jax.jit` and recompiles.

    Parameters
    ----------
    params : pytree
        The cosmological parameters, in any layout: a dict, a named tuple, a
        dataclass registered as a pytree.
    hubble : callable
        ``hubble(params, z)``, the Hubble rate :math:`H(z)` in 1/Mpc
        (:math:`c = 1`) with the shape of ``z``. Static.
    omega_m : callable
        ``omega_m(params)``, the matter density parameter today,
        :math:`\\Omega_m`. The matter is assumed to dilute as
        :math:`(1 + z)^3`. Static.
    omega_k : callable, optional
        ``omega_k(params)``, the curvature density parameter today,
        :math:`\\Omega_k`, positive for an open universe. ``None`` means a
        flat universe. Static.

    Examples
    --------
    >>> def hubble(p, z):
    ...     e2 = p["omega_m"] * (1 + z) ** 3 + 1 - p["omega_m"]
    ...     return p["h"] / 2997.92458 * jnp.sqrt(e2)
    >>> def omega_m(p):
    ...     return p["omega_m"]
    >>> background = GenericBackground({"h": 0.7, "omega_m": 0.3}, hubble, omega_m)
    """

    params: P
    hubble: Callable[[P, Array], Array]
    omega_m: Callable[[P], ArrayLike]
    omega_k: Callable[[P], ArrayLike] | None = None

    @property
    def hubble_constant(self) -> Float[Array, ""]:
        """Hubble constant :math:`H_0`.

        Returns
        -------
        Array
            ``hubble(params, 0)`` in 1/Mpc (:math:`c = 1`), scalar.
        """
        return self.hubble(self.params, jnp.zeros(()))

    def hubble_parameter(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Hubble rate :math:`H(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            ``hubble(params, z)`` in 1/Mpc, with the shape of ``z``.
        """
        return self.hubble(self.params, jnp.asarray(z))

    def matter_density_parameter(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Matter density parameter :math:`\\Omega_m(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`\\Omega_m (1 + z)^3 H_0^2 / H^2(z)`, with the shape of ``z``.
        """
        return functional.matter_density_parameter(
            self.hubble, self.params, z, self.omega_m(self.params)
        )

    def comoving_distance(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Line-of-sight comoving distance :math:`\\chi(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`\\chi(z)` in Mpc, with the shape of ``z``.
        """
        return functional.comoving_distance(self.hubble, self.params, z)

    def transverse_comoving_distance(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Transverse comoving distance :math:`D_M(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`D_M(z)` in Mpc, with the shape of ``z``; equal to
            :math:`\\chi(z)` when ``omega_k`` is ``None``.
        """
        if self.omega_k is None:
            return self.comoving_distance(z)
        return functional.transverse_comoving_distance(
            self.hubble, self.params, z, self.omega_k(self.params)
        )

    def redshift_at_comoving_distance(
        self,
        chi: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Redshift at a given comoving distance.

        Parameters
        ----------
        chi : ArrayLike
            Comoving distance in Mpc, non-negative and corresponding to a
            finite redshift.

        Returns
        -------
        Array
            Redshift :math:`z(\\chi)`, with the shape of ``chi``.
        """
        return functional.redshift_at_comoving_distance(self.hubble, self.params, chi)
