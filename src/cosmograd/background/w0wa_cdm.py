"""Curved background with CPL dark energy."""

from dataclasses import dataclass
from functools import partial

import jax
from jax.typing import ArrayLike
from jaxtyping import Real

from ..constants import T_CMB_FIDUCIAL
from ._base import CPLBackground

__all__ = ["W0WaCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "omega_k", "w0", "wa"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class W0WaCDM(CPLBackground):
    """Background with radiation, curvature and CPL dark energy.

    The dark energy equation of state is
    :math:`w(a) = w_0 + w_a (1 - a)` (Chevallier & Polarski 2001; Linder
    2003), whose density evolves in closed form, so the expansion rate is

    .. math::

        E^2(z) = \\Omega_m (1 + z)^3 + \\Omega_r (1 + z)^4
        + \\Omega_k (1 + z)^2
        + \\Omega_\\mathrm{de} (1 + z)^{3 (1 + w_0 + w_a)}
        \\exp\\!\\left(-\\frac{3 w_a z}{1 + z}\\right),

    with :math:`\\Omega_\\mathrm{de} = 1 - \\Omega_m - \\Omega_r - \\Omega_k`.
    Radiation is photons plus ``n_eff`` massless neutrino species, fixed by
    ``t_cmb``. Massive neutrinos are not included. Dark energy is smooth: it
    enters the expansion only.

    This is the most general background; :class:`~cosmograd.LCDM`,
    :class:`~cosmograd.FlatW0WaCDM` and :class:`~cosmograd.FlatLCDM` fix some
    of its parameters.

    The instance is a pytree whose leaves are ``H0``, ``omega_m``,
    ``omega_k``, ``w0`` and ``wa``; every other field is static. Methods
    operate on one cosmology with scalar parameters; evaluate a batch of
    cosmologies with :func:`jax.vmap`.

    Distances use 32-point Gauss–Legendre quadrature in
    :math:`\\ln(1 + z)`. The inverse relation uses a fixed number of Newton
    iterations in the same coordinate, with implicit differentiation through
    :func:`jax.lax.custom_root` and an analytic residual derivative. The
    inverse is of the line-of-sight distance :math:`\\chi`, not of the
    transverse distance.

    Parameters
    ----------
    H0 : ArrayLike
        Hubble constant in km/s/Mpc. Assumed positive.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m`. Assumed to lie in
        ``(0, 1 - Omega_r - Omega_k)`` so that :math:`\\Omega_\\mathrm{de} > 0`.
    omega_k : ArrayLike, default 0.0
        Curvature density parameter today, :math:`\\Omega_k`; positive for an
        open universe.
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

    # Tells jaxtyping's import hook not to check the generated ``__init__``
    # (methods and properties stay checked). JAX rebuilds this pytree with
    # non-array placeholder leaves during transforms such as ``jacfwd``, which
    # would fail the field annotations below.
    __no_type_check__ = True

    H0: Real[ArrayLike, "..."]
    omega_m: Real[ArrayLike, "..."]
    omega_k: Real[ArrayLike, "..."] = 0.0
    w0: Real[ArrayLike, "..."] = -1.0
    wa: Real[ArrayLike, "..."] = 0.0
    t_cmb: float = T_CMB_FIDUCIAL
    n_eff: float = 3.044
