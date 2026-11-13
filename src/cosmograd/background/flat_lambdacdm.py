"""Flat background with a cosmological constant."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..constants import T_CMB_FIDUCIAL
from ._base import _Flat, _Lambda

__all__ = ["FlatLCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class FlatLCDM(_Flat, _Lambda):
    """Flat :math:`\\Lambda\\mathrm{CDM}` background with radiation.

    This is :class:`~cosmograd.W0WaCDM` with :math:`\\Omega_k = 0`,
    :math:`w_0 = -1` and :math:`w_a = 0` fixed, so

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

    def efunc(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Dimensionless expansion rate :math:`E(z) = H(z) / H_0`.

        Drops the zero curvature term of the general form, which XLA does not
        fold away and which costs about a quarter of the distance evaluation.

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
