"""Linear and nonlinear power spectra of one bacco cosmology at one redshift.

Pure ``jax.numpy`` ports of ``baccoemu.Matter_powerspectrum.get_linear_pk`` and
``get_nonlinear_pk`` with their default settings (no-wiggles BAO smearing, no
baryonic boost), for both cold and total matter. Wavenumbers are in h/Mpc and
spectra in (Mpc/h)^3, as in ``baccoemu``.

The nonlinear pipeline:

1. Linear emulator: P_lin(k) on ``k_lin``.
2. No-wiggles emulator: P_nw = ratio * P_lin, the ratio forced to 1 for
   k > 0.8 h/Mpc.
3. Smeared BAO: P_smeared = G P_lin + (1 - G) P_nw with
   G = exp(-k^2 sigma_*^2 / 2) and sigma_*^2 = int k P_lin dlnk / (3 pi^2).
4. Nonlinear emulator: P_nl,cold = Q P_smeared on ``k_nl``. For total matter,
   P_nl,tot = (1 - f_nu)^2 (P_nl,cold - P_lin,cold) + P_lin,tot.
5. The boost P_nl / P_lin is interpolated linearly in ln k and set to 1 below
   ``k_nl[0]``; it multiplies P_lin interpolated in ln k - ln P.
"""

from collections.abc import Mapping

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ._emulator import (
    LINEAR_KEYS,
    NONLINEAR_KEYS,
    SIGMA8_KEYS,
    Emulator,
    mlp_apply,
)

__all__ = [
    "AS_REF",
    "NU_DENSITY",
    "Params",
    "amplitudes",
    "interp_extrap",
    "linear_log_pk",
    "nonlinear_boost",
]

Params = Mapping[str, Float[Array, ""]]
"""Scalar parameters of one cosmology, keyed by ``baccoemu`` name."""

AS_REF = 1e-9
"""Primordial amplitude at which the linear and sigma8 networks are trained."""
K_PIVOT = 0.05
"""Pivot scale of ``A_s`` and ``ns`` in 1/Mpc."""
NU_DENSITY = 93.14
"""``omega_nu h^2 = neutrino_mass / NU_DENSITY``, with the mass in eV."""
_K_NOWIGGLE_MAX = 0.8  # h/Mpc, above which P_nw = P_lin


def interp_extrap(
    x: Float[Array, "*shape"],
    xp: Float[Array, " n"],
    fp: Float[Array, " n"],
) -> Float[Array, "*shape"]:
    """Piecewise-linear interpolation with linear extrapolation at both ends.

    Mirrors ``scipy.interpolate.interp1d(kind="linear", fill_value="extrapolate")``.

    Parameters
    ----------
    x : Array
        Points at which to evaluate.
    xp : Array
        Strictly increasing abscissae, at least two points.
    fp : Array
        Values at ``xp``.

    Returns
    -------
    Array
        Interpolated or extrapolated values, with the shape of ``x``.
    """
    inner = jnp.interp(x, xp, fp)
    lo = fp[0] + (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x - xp[0])
    hi = fp[-1] + (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x - xp[-1])
    return jnp.where(x < xp[0], lo, jnp.where(x > xp[-1], hi, inner))


def _stack(p: Params, keys: tuple[str, ...]) -> Float[Array, " n"]:
    return jnp.stack([p[key] for key in keys])


def amplitudes(emu: Emulator, p: Params) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Convert between ``A_s`` and ``sigma8_cold``.

    Parameters
    ----------
    emu : Emulator
        Loaded emulator.
    p : Params
        Every key of ``SIGMA8_KEYS`` and exactly one of ``A_s`` and
        ``sigma8_cold``.

    Returns
    -------
    A_s, sigma8_cold : Array
        Both amplitudes, scalars. ``sigma8_cold`` is at z = 0.
    """
    sigma8_ref = mlp_apply(emu.sigma8, _stack(p, SIGMA8_KEYS))[0]
    if "A_s" in p:
        return p["A_s"], sigma8_ref * jnp.sqrt(p["A_s"] / AS_REF)
    return AS_REF * (p["sigma8_cold"] / sigma8_ref) ** 2, p["sigma8_cold"]


def linear_log_pk(emu: Emulator, p: Params, *, cold: bool) -> Float[Array, " k_lin"]:
    """Natural log of the linear power spectrum on ``emu.k_lin``.

    Parameters
    ----------
    emu : Emulator
        Loaded emulator.
    p : Params
        Every key of ``LINEAR_KEYS`` plus ``A_s`` and ``ns``.
    cold : bool
        Cold (CDM + baryons) instead of total matter.

    Returns
    -------
    Array
        :math:`\\ln P_\\mathrm{lin}` with P in (Mpc/h)^3, shape ``(n_k_lin,)``.
    """
    mlp, mean = (
        (emu.linear_cold, emu.linear_cold_mean)
        if cold
        else (emu.linear_total, emu.linear_total_mean)
    )
    log_pk = mlp_apply(mlp, _stack(p, LINEAR_KEYS)) + mean
    log_tilt = (p["ns"] - 1.0) * jnp.log(emu.k_lin * p["hubble"] / K_PIVOT)
    return log_pk + jnp.log(p["A_s"] / AS_REF) + log_tilt


def nonlinear_boost(
    emu: Emulator,
    p: Params,
    log_pk_cold: Float[Array, " k_lin"],
    log_pk: Float[Array, " k_lin"],
    *,
    cold: bool,
) -> Float[Array, " k_nl"]:
    """Nonlinear-to-linear ratio :math:`P_\\mathrm{nl} / P_\\mathrm{lin}` on ``emu.k_nl``.

    Parameters
    ----------
    emu : Emulator
        Loaded emulator.
    p : Params
        Every key of ``NONLINEAR_KEYS``.
    log_pk_cold : Array
        Linear cold-matter :func:`linear_log_pk` for the same cosmology.
    log_pk : Array
        Linear :func:`linear_log_pk` of the requested species; the same as
        ``log_pk_cold`` when ``cold`` is true.
    cold : bool
        Cold (CDM + baryons) instead of total matter.

    Returns
    -------
    Array
        The boost, shape ``(n_k_nl,)``.
    """
    log_k_lin = jnp.log(emu.k_lin)
    log_k_nl = jnp.log(emu.k_nl)
    pk_cold = jnp.exp(log_pk_cold)
    pk_cold_nl = jnp.exp(interp_extrap(log_k_nl, log_k_lin, log_pk_cold))

    # No-wiggles spectrum (same inputs and k grid as the linear network).
    log_ratio = mlp_apply(emu.no_wiggles, _stack(p, LINEAR_KEYS)) + emu.no_wiggles_mean
    log_ratio = jnp.where(emu.k_lin > _K_NOWIGGLE_MAX, 0.0, log_ratio)
    ratio_nl = jnp.exp(jnp.interp(log_k_nl, log_k_lin, log_ratio, left=0.0, right=0.0))

    # BAO damping by the linear displacement dispersion.
    sigma_star_2 = jnp.trapezoid(emu.k_lin * pk_cold, x=log_k_lin) / (3 * jnp.pi**2)
    damping = jnp.exp(-0.5 * emu.k_nl**2 * sigma_star_2)
    pk_smeared = pk_cold_nl * (damping + ratio_nl * (1 - damping))

    lo, hi = emu.bounds_nonlinear[:, 0], emu.bounds_nonlinear[:, 1]
    x_nl = (_stack(p, NONLINEAR_KEYS) - lo) / (hi - lo)
    q = jnp.exp(mlp_apply(emu.nonlinear, x_nl) + emu.nonlinear_mean)
    if cold:
        return q * pk_smeared / pk_cold_nl

    omega_nu = p["neutrino_mass"] / NU_DENSITY / p["hubble"] ** 2
    f_nu = omega_nu / (p["omega_cold"] + omega_nu)
    pk_nl = jnp.exp(interp_extrap(log_k_nl, log_k_lin, log_pk))
    return ((1 - f_nu) ** 2 * (q * pk_smeared - pk_cold_nl) + pk_nl) / pk_nl
