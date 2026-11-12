"""Native JAX background cosmologies."""

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
from .protocol import Background

__all__ = ["FlatLCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["h", "omega_m"],
    meta_fields=["t_cmb", "n_eff", "z_max", "n_grid"],
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

    The instance is a pytree whose leaves are ``h`` and ``omega_m``; every
    other field is static. Differentiate with respect to the cosmology
    directly, e.g. ``jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)``.

    Methods operate on one cosmology with scalar parameters. Parameter fields
    allow arbitrary shapes to hold batched pytrees and Jacobians; evaluate a
    batch of cosmologies with :func:`jax.vmap`.

    Distances come from a table of :math:`\\chi` on a grid uniform in
    :math:`\\ln(1 + z)` over ``[0, z_max]``, rebuilt from the parameters on
    every call. Inside one jitted function, evaluate once on your own grid and
    reuse the array.

    Parameters
    ----------
    h : ArrayLike
        Dimensionless Hubble constant, :math:`H_0 = 100\\,h` km/s/Mpc. Assumed
        positive.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m`. Assumed to lie in
        ``(0, 1 - Omega_r)`` so that :math:`\\Omega_\\Lambda > 0`.
    t_cmb : float, default 2.7255
        CMB temperature today in K. Static.
    n_eff : float, default 3.044
        Effective number of massless neutrino species. Static.
    z_max : float, default 1200.0
        Largest redshift of the distance table. Static. Distances are valid
        for ``0 <= z <= z_max``; outside that range the table is clamped, so
        results there are wrong without warning.
    n_grid : int, default 2048
        Number of points in the distance table. Static.

    Notes
    -----
    No method validates its inputs. Redshifts are expected to be non-negative
    and at most ``z_max``; comoving distances at most
    ``comoving_distance(z_max)``. They need not be sorted.
    """

    h: Real[ArrayLike, "..."]
    omega_m: Real[ArrayLike, "..."]
    t_cmb: float = T_CMB_FIDUCIAL
    n_eff: float = 3.044
    z_max: float = 1200.0
    n_grid: int = 2048

    @property
    def omega_r(self) -> Float[Array, ""]:
        """Radiation density parameter today, :math:`\\Omega_r`.

        Returns
        -------
        Array
            Photons plus ``n_eff`` massless neutrinos, dimensionless, scalar.
        """
        omega_gamma_h2 = OMEGA_GAMMA_H2_FIDUCIAL * (self.t_cmb / T_CMB_FIDUCIAL) ** 4
        omega_gamma = omega_gamma_h2 / jnp.asarray(self.h) ** 2
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
        return 100 * jnp.asarray(self.h) / C_KM_S

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

    def _distance_table(self) -> tuple[Float[Array, " grid"], Float[Array, " grid"]]:
        """Tabulate :math:`\\chi` on a grid uniform in :math:`\\ln(1 + z)`.

        Returns
        -------
        x : Array
            Grid of :math:`\\ln(1 + z)`, shape ``(n_grid,)``.
        chi : Array
            Comoving distance in Mpc at each grid point, shape ``(n_grid,)``,
            from the cumulative trapezoid rule on
            :math:`\\mathrm{d}\\chi / \\mathrm{d}x = (1 + z) / H(z)`.
        """
        x = jnp.linspace(0.0, jnp.log1p(self.z_max), self.n_grid)
        zp1 = jnp.exp(x)
        integrand = zp1 / self.hubble_parameter(zp1 - 1)
        steps = 0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(x)
        chi = jnp.concatenate([jnp.zeros(1, dtype=steps.dtype), jnp.cumsum(steps)])
        return x, chi

    def _distance_ratio_table(
        self,
    ) -> tuple[Float[Array, " grid"], Float[Array, " grid"]]:
        """Tabulate the ratio :math:`\\chi / \\ln(1 + z)`.

        The ratio is smooth and tends to :math:`1 / H_0` as :math:`z \\to 0`, so
        interpolating it linearly keeps the relative error of :math:`\\chi`
        small at low redshift, where interpolating :math:`\\chi` itself does
        not.

        Returns
        -------
        x : Array
            Grid of :math:`\\ln(1 + z)`, shape ``(n_grid,)``.
        ratio : Array
            :math:`\\chi / x` in Mpc, shape ``(n_grid,)``, with the
            :math:`x \\to 0` limit :math:`1 / H_0` at the first point.
        """
        x, chi = self._distance_table()
        ratio = jnp.concatenate([(1 / self.hubble_constant)[None], chi[1:] / x[1:]])
        return x, ratio

    def comoving_distance(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Line-of-sight comoving distance :math:`\\chi(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift in ``[0, z_max]``. Values outside are clamped to the
            table's ends.

        Returns
        -------
        Array
            :math:`\\chi(z)` in Mpc, with the shape of ``z``.
        """
        x, ratio = self._distance_ratio_table()
        x_eval = jnp.log1p(jnp.asarray(z))
        return x_eval * jnp.interp(x_eval, x, ratio)

    def redshift_at_comoving_distance(
        self,
        chi: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Redshift at a given comoving distance.

        Parameters
        ----------
        chi : ArrayLike
            Comoving distance in Mpc, in ``[0, comoving_distance(z_max)]``.
            Values outside are clamped to the table's ends.

        Returns
        -------
        Array
            Redshift :math:`z(\\chi)`, with the shape of ``chi``.
        """
        x, ratio = self._distance_ratio_table()
        chi = jnp.asarray(chi)
        return jnp.expm1(chi / jnp.interp(chi, x * ratio, ratio))
