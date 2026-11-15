"""Flat background with a cosmological constant."""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..constants import (
    LN10AS_FIDUCIAL,
    N_S_FIDUCIAL,
    OMEGA_B_FIDUCIAL,
    T_CMB_FIDUCIAL,
)
from ._base import _Flat, _given, _Lambda

__all__ = ["FlatLCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "omega_b", "n_s", "ln10as"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class FlatLCDM(_Flat, _Lambda):
    """Flat :math:`\\Lambda\\mathrm{CDM}` background with radiation.

    Massive neutrinos are not included.

    The instance is a pytree whose leaves are ``H0``, ``omega_m`` and ``omega_b``,
    ``n_s`` and ``ln10as``; ``t_cmb`` and ``n_eff`` are static.
    Differentiate with respect to the cosmology directly, e.g.
    ``jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)``.

    ``omega_b``, ``n_s`` and ``ln10as`` are read only by the perturbation
    backends, which take this object as their cosmology; no method here uses
    them. They default to the Planck 2018 values of
    :data:`~cosmograd.cosmologies.Planck18` and are always pytree leaves, so a
    backend given a cosmology that leaves them out uses those values without
    warning. The default ``omega_b`` does not follow ``H0``; set it explicitly
    to hold :math:`\\Omega_b h^2` fixed.

    Parameters
    ----------
    H0 : ArrayLike
        Hubble constant in km/s/Mpc. Assumed positive.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m`. Assumed to lie in
        ``(0, 1 - Omega_r)`` so that :math:`\\Omega_\\Lambda > 0`.
    omega_b : ArrayLike, default 0.04897
        Baryon density parameter today, :math:`\\Omega_b`, part of
        ``omega_m``.
    n_s : ArrayLike, default 0.9665
        Scalar spectral index.
    ln10as : ArrayLike, default 3.047
        :math:`\\ln(10^{10} A_s)`, with :math:`A_s` the primordial curvature
        amplitude at the pivot 0.05/Mpc.
    t_cmb : float, default 2.7255
        CMB temperature today in K. Static.
    n_eff : float, default 3.044
        Effective number of massless neutrino species. Static.
    """

    # Tells jaxtyping's import hook not to check the generated ``__init__``
    # (methods and properties stay checked). JAX rebuilds this pytree with
    # non-array placeholder leaves during transforms such as ``jacfwd``, which
    # would fail the field annotations below.
    __no_type_check__ = True

    H0: Real[ArrayLike, "..."]
    omega_m: Real[ArrayLike, "..."]
    omega_b: Real[ArrayLike, "..."] = OMEGA_B_FIDUCIAL
    n_s: Real[ArrayLike, "..."] = N_S_FIDUCIAL
    ln10as: Real[ArrayLike, "..."] = LN10AS_FIDUCIAL
    t_cmb: float = T_CMB_FIDUCIAL
    n_eff: float = 3.044

    def with_parameters(
        self,
        *,
        H0: Real[ArrayLike, "..."] | None = None,
        omega_m: Real[ArrayLike, "..."] | None = None,
        omega_b: Real[ArrayLike, "..."] | None = None,
        n_s: Real[ArrayLike, "..."] | None = None,
        ln10as: Real[ArrayLike, "..."] | None = None,
        t_cmb: float | None = None,
        n_eff: float | None = None,
    ) -> "FlatLCDM":
        """Return a copy with the given parameters replaced.

        Parameters left as ``None`` keep their current value. The signature
        lists only the parameters of this model, so a parameter the model
        fixes is rejected by static checkers.

        Returns
        -------
        FlatLCDM
            A new instance; ``self`` is unchanged.
        """
        return replace(
            self,
            **_given(
                H0=H0,
                omega_m=omega_m,
                omega_b=omega_b,
                n_s=n_s,
                ln10as=ln10as,
                t_cmb=t_cmb,
                n_eff=n_eff,
            ),
        )

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
