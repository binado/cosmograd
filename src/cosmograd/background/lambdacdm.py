"""Curved background with a cosmological constant."""

from dataclasses import dataclass, replace
from functools import partial

import jax
from jax.typing import ArrayLike
from jaxtyping import Real

from ..constants import T_CMB_FIDUCIAL
from ._base import _given, _Lambda

__all__ = ["LCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "omega_k"],
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

    The instance is a pytree whose leaves are ``H0``, ``omega_m`` and
    ``omega_k``; every other field is static. Methods operate on one
    cosmology with scalar parameters; evaluate a batch of cosmologies with
    :func:`jax.vmap`.

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
    t_cmb: float = T_CMB_FIDUCIAL
    n_eff: float = 3.044

    def with_parameters(
        self,
        *,
        H0: Real[ArrayLike, "..."] | None = None,
        omega_m: Real[ArrayLike, "..."] | None = None,
        omega_k: Real[ArrayLike, "..."] | None = None,
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
            **_given(H0=H0, omega_m=omega_m, omega_k=omega_k, t_cmb=t_cmb, n_eff=n_eff),
        )
