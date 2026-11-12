"""Native JAX background cosmologies."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..constants import (
    C_KM_S,
    NEUTRINO_TO_PHOTON_PER_SPECIES,
    OMEGA_GAMMA_H2_FIDUCIAL,
    T_CMB_FIDUCIAL,
)
from ..utils import (
    affine_transform_gauss_legendre_nodes,
    gauss_legendre_nodes_and_weights,
)
from .protocol import Background

__all__ = ["FlatLCDM"]

# Fixed number of Newton iterations used by the inverse distance relation.
_NEWTON_ITERATIONS = 7


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class FlatLCDM(Background):
    """Flat :math:`\\Lambda\\mathrm{CDM}` background with radiation.

    The expansion rate is

    .. math::

        E^2(z) = \\Omega_m (1 + z)^3 + \\Omega_r (1 + z)^4 + \\Omega_\\Lambda,

    with :math:`\\Omega_\\Lambda = 1 - \\Omega_m - \\Omega_r`. Radiation is
    photons plus ``n_eff`` massless neutrino species, fixed by ``t_cmb``.
    Massive neutrinos are not included.

    The instance is a pytree whose leaves are ``H0`` and ``omega_m``; every
    other field is static. Differentiate with respect to the cosmology
    directly, e.g. ``jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)``.

    Methods operate on one cosmology with scalar parameters. Parameter fields
    allow arbitrary shapes to hold batched pytrees and Jacobians; evaluate a
    batch of cosmologies with :func:`jax.vmap`.

    Distances use 32-point Gauss–Legendre quadrature in
    :math:`\\ln(1 + z)`. The inverse relation uses a fixed number of Newton
    iterations in the same coordinate, with implicit differentiation through
    :func:`jax.lax.custom_root` and an analytic residual derivative.

    Parameters
    ----------
    H0 : ArrayLike
        Hubble constant in km/s/Mpc. Assumed positive.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m`. Assumed to lie in
        ``(0, 1 - Omega_r)`` so that :math:`\\Omega_\\Lambda > 0`.
    t_cmb : float, default 2.7255
        CMB temperature today in K. Static.
    n_eff : float, default 3.044
        Effective number of massless neutrino species. Static.

    Notes
    -----
    No method validates its inputs. Redshifts and distances are expected to be
    non-negative. Quadrature accuracy and convergence of the inverse have been
    checked through ``z = 1200`` for representative cosmologies. The fixed
    iteration budget does not guarantee convergence for arbitrary parameters
    or distances. Inputs need not be sorted.
    """

    # Tells jaxtyping's import hook not to check the generated ``__init__``
    # (methods and properties stay checked). JAX rebuilds this pytree with
    # non-array placeholder leaves during transforms such as ``jacfwd``, which
    # would fail the field annotations below.
    __no_type_check__ = True

    H0: Real[ArrayLike, "..."]
    omega_m: Real[ArrayLike, "..."]
    t_cmb: float = T_CMB_FIDUCIAL
    n_eff: float = 3.044

    @property
    def omega_r(self) -> Float[Array, ""]:
        """Radiation density parameter today, :math:`\\Omega_r`.

        Returns
        -------
        Array
            Photons plus ``n_eff`` massless neutrinos, dimensionless, scalar.
        """
        omega_gamma_h2 = OMEGA_GAMMA_H2_FIDUCIAL * (self.t_cmb / T_CMB_FIDUCIAL) ** 4
        omega_gamma = omega_gamma_h2 / (jnp.asarray(self.H0) / 100) ** 2
        return omega_gamma * (1 + NEUTRINO_TO_PHOTON_PER_SPECIES * self.n_eff)

    @property
    def omega_de(self) -> Float[Array, ""]:
        """Dark energy density parameter today, :math:`\\Omega_\\Lambda`.

        Returns
        -------
        Array
            :math:`1 - \\Omega_m - \\Omega_r`, dimensionless, scalar.
        """
        return 1 - jnp.asarray(self.omega_m) - self.omega_r

    def efunc(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Dimensionless expansion rate :math:`E(z) = H(z) / H_0`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`E(z)`, with the shape of ``z``.
        """
        zp1 = 1 + jnp.asarray(z)
        return jnp.sqrt(self.omega_m * zp1**3 + self.omega_r * zp1**4 + self.omega_de)

    @property
    def hubble_constant(self) -> Float[Array, ""]:
        """Hubble constant :math:`H_0`.

        Returns
        -------
        Array
            :math:`H_0` in 1/Mpc (:math:`c = 1`), scalar.
        """
        return jnp.asarray(self.H0) / C_KM_S

    def hubble_parameter(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Hubble rate :math:`H(z) = H_0 E(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`H(z)` in 1/Mpc, with the shape of ``z``.
        """
        return self.hubble_constant * self.efunc(z)

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
            :math:`\\Omega_m (1 + z)^3 / E^2(z)`, with the shape of ``z``.
        """
        zp1 = 1 + jnp.asarray(z)
        return self.omega_m * zp1**3 / self.efunc(z) ** 2

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
        with jax.ensure_compile_time_eval():
            nodes, weights = gauss_legendre_nodes_and_weights(32)
        length = jnp.log1p(jnp.asarray(z))
        x = affine_transform_gauss_legendre_nodes(nodes, jnp.zeros_like(length), length)
        integrand = jnp.exp(x) / self.hubble_parameter(jnp.expm1(x))
        return 0.5 * length * jnp.sum(weights * integrand, axis=-1)

    def redshift_at_comoving_distance(
        self,
        chi: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Redshift at a given comoving distance.

        Parameters
        ----------
        chi : ArrayLike
            Comoving distance in Mpc, non-negative and corresponding to a
            finite redshift within the documented convergence range.

        Returns
        -------
        Array
            Redshift :math:`z(\\chi)`, with the shape of ``chi``.
        """
        # Review this implementation if inverse distances become a performance
        # bottleneck: a custom JVP on the inverse can reduce gradient overhead.
        chi = jnp.asarray(chi, dtype=jnp.result_type(chi, 1.0))

        def residual(x: Array) -> Array:
            return _distance_residual(self, chi, x)

        def solve(f: Callable[[Array], Array], initial: Array) -> Array:
            def step(_: int | Array, x: Array) -> Array:
                z = jnp.expm1(x)
                return x - f(x) * self.hubble_parameter(z) / jnp.exp(x)

            return jax.lax.fori_loop(0, _NEWTON_ITERATIONS, step, initial)

        def tangent_solve(g: Callable[[Array], Array], y: Array) -> Array:
            # Each distance has an independent root, so the Jacobian is diagonal.
            return y / g(jnp.ones_like(y))

        x = jax.lax.custom_root(
            residual, chi * self.hubble_constant, solve, tangent_solve
        )
        return jnp.expm1(x)


@jax.custom_jvp
def _distance_residual(
    cosmo: FlatLCDM,
    chi: Real[ArrayLike, "*shape"],
    x: Real[ArrayLike, "*shape"],
) -> Float[Array, "*shape"]:
    """Distance residual in log-redshift for independent scalar roots."""
    return cosmo.comoving_distance(jnp.expm1(x)) - jnp.asarray(chi)


@_distance_residual.defjvp
def _distance_residual_jvp(
    primals: tuple[FlatLCDM, Array, Array],
    tangents: tuple[FlatLCDM, Array, Array],
) -> tuple[Array, Array]:
    """Use dchi/dx = exp(x)/H(z), retaining parameter derivatives of chi."""
    cosmo, chi, x = primals
    dcosmo, dchi, dx = tangents
    z = jnp.expm1(x)
    value, dparameter = jax.jvp(lambda c: c.comoving_distance(z), (cosmo,), (dcosmo,))
    derivative = dparameter - dchi + jnp.exp(x) / cosmo.hubble_parameter(z) * dx
    return value - chi, derivative
