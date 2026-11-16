"""Machinery shared by every background with radiation and CPL dark energy."""

from typing import TYPE_CHECKING

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
from . import functional
from .protocol import Background

__all__ = ["CPLBackground"]


def _given(**changes: object) -> dict[str, object]:
    """Drop the entries of ``changes`` that are ``None``."""
    return {k: v for k, v in changes.items() if v is not None}


class CPLBackground(Background):
    """Background with matter, radiation, curvature and CPL dark energy."""

    # Declared as read-only properties for static checkers only: subclasses
    # are frozen dataclasses, whose fields cannot override writable attributes.
    if TYPE_CHECKING:

        @property
        def H0(self) -> Real[ArrayLike, "..."]: ...  # noqa: N802
        @property
        def omega_m(self) -> Real[ArrayLike, "..."]: ...
        @property
        def omega_k(self) -> Real[ArrayLike, "..."]: ...
        @property
        def w0(self) -> Real[ArrayLike, "..."]: ...
        @property
        def wa(self) -> Real[ArrayLike, "..."]: ...
        @property
        def t_cmb(self) -> float: ...
        @property
        def n_eff(self) -> float: ...

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
        """Dark energy density parameter today, :math:`\\Omega_\\mathrm{de}`.

        Returns
        -------
        Array
            :math:`1 - \\Omega_m - \\Omega_r - \\Omega_k`, dimensionless,
            scalar.
        """
        return 1 - jnp.asarray(self.omega_m) - self.omega_r - jnp.asarray(self.omega_k)

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
        z = jnp.asarray(z)
        zp1 = 1 + z
        w0, wa = jnp.asarray(self.w0), jnp.asarray(self.wa)
        dark_energy = zp1 ** (3 * (1 + w0 + wa)) * jnp.exp(-3 * wa * z / zp1)
        return jnp.sqrt(
            self.omega_m * zp1**3
            + self.omega_r * zp1**4
            + jnp.asarray(self.omega_k) * zp1**2
            + self.omega_de * dark_energy
        )

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
        return functional.comoving_distance(type(self).hubble_parameter, self, z)

    def transverse_comoving_distance(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Transverse comoving distance :math:`D_M(z)`.

        With :math:`x = \\Omega_k (H_0 \\chi)^2` this is
        :math:`\\chi\\, \\mathrm{sinhc}(\\sqrt{x})`, using
        :math:`\\sin` for negative :math:`x`. The expression is smooth and
        differentiable through :math:`\\Omega_k = 0`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`D_M(z)` in Mpc, with the shape of ``z``.
        """
        return functional.transverse_comoving_distance(
            type(self).hubble_parameter, self, z, self.omega_k
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
            finite redshift within the documented convergence range.

        Returns
        -------
        Array
            Redshift :math:`z(\\chi)`, with the shape of ``chi``.
        """
        return functional.redshift_at_comoving_distance(
            type(self).hubble_parameter, self, chi
        )


class _Flat(CPLBackground):
    """Mixin fixing :math:`\\Omega_k = 0`.

    Not public. Flat models have no curvature parameter and skip the
    curvature correction in :meth:`transverse_comoving_distance`.
    """

    @property
    def omega_k(self) -> float:
        """Curvature density parameter, identically zero."""
        return 0.0

    def transverse_comoving_distance(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Transverse comoving distance, :math:`D_M = \\chi` when flat.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`D_M(z)` in Mpc, with the shape of ``z``.
        """
        return self.comoving_distance(z)


class _Lambda(CPLBackground):
    """Mixin fixing the cosmological constant, :math:`w_0 = -1, w_a = 0`.

    Not public.
    """

    @property
    def w0(self) -> float:
        """Dark energy equation of state today, identically :math:`-1`."""
        return -1.0

    @property
    def wa(self) -> float:
        """Evolution of the equation of state, identically zero."""
        return 0.0
