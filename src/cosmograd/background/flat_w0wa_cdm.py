"""Flat background with CPL dark energy."""

from dataclasses import dataclass, replace
from functools import partial

import jax
from jax.typing import ArrayLike
from jaxtyping import Real

from ..constants import T_CMB_FIDUCIAL
from ._base import _Flat, _given

__all__ = ["FlatW0WaCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "w0", "wa"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class FlatW0WaCDM(_Flat):
    """Flat background with radiation and CPL dark energy.

    This is :class:`~cosmograd.W0WaCDM` with :math:`\\Omega_k = 0` fixed. The
    dark energy equation of state is :math:`w(a) = w_0 + w_a (1 - a)`
    (Chevallier & Polarski 2001; Linder 2003), whose density evolves in
    closed form, so the expansion rate is

    .. math::

        E^2(z) = \\Omega_m (1 + z)^3 + \\Omega_r (1 + z)^4
        + \\Omega_\\mathrm{de} (1 + z)^{3 (1 + w_0 + w_a)}
        \\exp\\!\\left(-\\frac{3 w_a z}{1 + z}\\right),

    with :math:`\\Omega_\\mathrm{de} = 1 - \\Omega_m - \\Omega_r`. Radiation is
    photons plus ``n_eff`` massless neutrino species, fixed by ``t_cmb``.
    Massive neutrinos are not included. Dark energy is smooth: it enters the
    expansion only. At the defaults ``w0 = -1`` and ``wa = 0`` this is
    :class:`~cosmograd.FlatLCDM`.

    The instance is a pytree whose leaves are ``H0``, ``omega_m``, ``w0`` and
    ``wa``; every other field is static. Methods operate on one cosmology
    with scalar parameters; evaluate a batch of cosmologies with
    :func:`jax.vmap`.

    Distances and their inverse are computed as in
    :class:`~cosmograd.FlatLCDM`.

    Parameters
    ----------
    H0 : ArrayLike
        Hubble constant in km/s/Mpc. Assumed positive.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m`. Assumed to lie in
        ``(0, 1 - Omega_r)`` so that :math:`\\Omega_\\mathrm{de} > 0`.
    w0 : ArrayLike, default -1.0
        Dark energy equation of state today.
    wa : ArrayLike, default 0.0
        Its derivative with respect to :math:`-a`.
    t_cmb : float, default 2.7255
        CMB temperature today in K. Static.
    n_eff : float, default 3.044
        Effective number of massless neutrino species. Static.

    Notes
    -----
    No method validates its inputs. Redshifts and distances are expected to be
    non-negative. The early-time equation of state is :math:`w_0 + w_a`; it
    should be negative, and well below zero for dark energy to be negligible
    in the matter era, which
    :func:`~cosmograd.perturbations.linear_growth` assumes for its initial
    conditions. The fixed iteration budget of the inverse does not guarantee
    convergence for arbitrary parameters or distances.
    """

    # See FlatLCDM: keep jaxtyping from checking the generated ``__init__``,
    # which JAX calls with placeholder leaves during transforms.
    __no_type_check__ = True

    H0: Real[ArrayLike, "..."]
    omega_m: Real[ArrayLike, "..."]
    w0: Real[ArrayLike, "..."] = -1.0
    wa: Real[ArrayLike, "..."] = 0.0
    t_cmb: float = T_CMB_FIDUCIAL
    n_eff: float = 3.044

    def with_parameters(
        self,
        *,
        H0: Real[ArrayLike, "..."] | None = None,
        omega_m: Real[ArrayLike, "..."] | None = None,
        w0: Real[ArrayLike, "..."] | None = None,
        wa: Real[ArrayLike, "..."] | None = None,
        t_cmb: float | None = None,
        n_eff: float | None = None,
    ) -> "FlatW0WaCDM":
        """Return a copy with the given parameters replaced.

        Parameters left as ``None`` keep their current value. The signature
        lists only the parameters of this model, so a parameter the model
        fixes is rejected by static checkers.

        Returns
        -------
        FlatW0WaCDM
            A new instance; ``self`` is unchanged.
        """
        return replace(
            self,
            **_given(H0=H0, omega_m=omega_m, w0=w0, wa=wa, t_cmb=t_cmb, n_eff=n_eff),
        )
