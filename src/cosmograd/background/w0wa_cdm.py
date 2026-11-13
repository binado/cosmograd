"""Curved background with CPL dark energy."""

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
from ._base import CPLBackground, _given

__all__ = ["W0WaCDM"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "omega_k", "w0", "wa", "omega_b", "n_s", "ln10as"],
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
    ``omega_k``, ``w0``, ``wa`` and ``omega_b``,
    ``n_s`` and ``ln10as``;
    ``t_cmb`` and ``n_eff`` are static. Methods operate on one cosmology with
    scalar parameters; evaluate a batch of cosmologies with :func:`jax.vmap`.

    Distances use 32-point Gauss–Legendre quadrature in
    :math:`\\ln(1 + z)`. The inverse relation uses a fixed number of Newton
    iterations in the same coordinate, with implicit differentiation through
    :func:`jax.lax.custom_root` and an analytic residual derivative. The
    inverse is of the line-of-sight distance :math:`\\chi`, not of the
    transverse distance.

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
        ``(0, 1 - Omega_r - Omega_k)`` so that :math:`\\Omega_\\mathrm{de} > 0`.
    omega_k : ArrayLike, default 0.0
        Curvature density parameter today, :math:`\\Omega_k`; positive for an
        open universe.
    w0 : ArrayLike, default -1.0
        Dark energy equation of state today.
    wa : ArrayLike, default 0.0
        Its derivative with respect to :math:`-a`.
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
        w0: Real[ArrayLike, "..."] | None = None,
        wa: Real[ArrayLike, "..."] | None = None,
        omega_b: Real[ArrayLike, "..."] | None = None,
        n_s: Real[ArrayLike, "..."] | None = None,
        ln10as: Real[ArrayLike, "..."] | None = None,
        t_cmb: float | None = None,
        n_eff: float | None = None,
    ) -> "W0WaCDM":
        """Return a copy with the given parameters replaced.

        Parameters left as ``None`` keep their current value. The signature
        lists only the parameters of this model, so a parameter the model
        fixes is rejected by static checkers.

        Returns
        -------
        W0WaCDM
            A new instance; ``self`` is unchanged.
        """
        return replace(
            self,
            **_given(
                H0=H0,
                omega_m=omega_m,
                omega_k=omega_k,
                w0=w0,
                wa=wa,
                omega_b=omega_b,
                n_s=n_s,
                ln10as=ln10as,
                t_cmb=t_cmb,
                n_eff=n_eff,
            ),
        )
