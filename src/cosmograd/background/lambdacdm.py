"""Curved background with a cosmological constant."""

from dataclasses import dataclass, replace
from functools import partial

import jax
from jax.typing import ArrayLike
from jaxtyping import Real

from ..constants import (
    LN10AS_FIDUCIAL,
    N_S_FIDUCIAL,
    OMEGA_B_FIDUCIAL,
    T_CMB_FIDUCIAL,
)
from ._base import _given, _Lambda

__all__ = ["LCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "omega_k", "omega_b", "n_s", "ln10as"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class LCDM(_Lambda):
    """:math:`\\Lambda\\mathrm{CDM}` background with radiation and curvature.

    This is :class:`~cosmograd.W0WaCDM` with :math:`w_0 = -1` and
    :math:`w_a = 0` fixed, so

    .. math::

        E^2(z) = \\Omega_m (1 + z)^3 + \\Omega_r (1 + z)^4
        + \\Omega_k (1 + z)^2 + \\Omega_\\Lambda,

    with :math:`\\Omega_\\Lambda = 1 - \\Omega_m - \\Omega_r - \\Omega_k`.
    Radiation is photons plus ``n_eff`` massless neutrino species, fixed by
    ``t_cmb``. Massive neutrinos are not included.

    The instance is a pytree whose leaves are ``H0``, ``omega_m``,
    ``omega_k`` and ``omega_b``,
    ``n_s`` and ``ln10as``; ``t_cmb`` and
    ``n_eff`` are static. Methods operate on one cosmology with scalar
    parameters; evaluate a batch of cosmologies with :func:`jax.vmap`.

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
        ``(0, 1 - Omega_r - Omega_k)`` so that :math:`\\Omega_\\Lambda > 0`.
    omega_k : ArrayLike, default 0.0
        Curvature density parameter today, :math:`\\Omega_k`; positive for an
        open universe.
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

    Notes
    -----
    No method validates its inputs. See :class:`~cosmograd.W0WaCDM` for the
    quadrature and inverse.
    """

    # See W0WaCDM: keep jaxtyping from checking the generated ``__init__``.
    __no_type_check__ = True

    H0: Real[ArrayLike, "..."]
    omega_m: Real[ArrayLike, "..."]
    omega_k: Real[ArrayLike, "..."] = 0.0
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
        omega_k: Real[ArrayLike, "..."] | None = None,
        omega_b: Real[ArrayLike, "..."] | None = None,
        n_s: Real[ArrayLike, "..."] | None = None,
        ln10as: Real[ArrayLike, "..."] | None = None,
        t_cmb: float | None = None,
        n_eff: float | None = None,
    ) -> "LCDM":
        """Return a copy with the given parameters replaced.

        Parameters left as ``None`` keep their current value. The signature
        lists only the parameters of this model, so a parameter the model
        fixes is rejected by static checkers.

        Returns
        -------
        LCDM
            A new instance; ``self`` is unchanged.
        """
        return replace(
            self,
            **_given(
                H0=H0,
                omega_m=omega_m,
                omega_k=omega_k,
                omega_b=omega_b,
                n_s=n_s,
                ln10as=ln10as,
                t_cmb=t_cmb,
                n_eff=n_eff,
            ),
        )
