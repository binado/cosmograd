"""Background quantities as pure functions of a Hubble rate.

Every function takes ``hubble_parameter(params, z)``, returning :math:`H(z)` in
1/Mpc (:math:`c = 1`) with the shape of ``z``, and the ``params`` it is called
with. ``params`` is any pytree, such as a dict, a named tuple or a dataclass,
and is the only thing differentiated or batched over; the functions never look
inside it. The Hubble constant is read as ``hubble_parameter(params, 0.0)``.

The native backgrounds are thin bindings of these functions, passing themselves
as ``params``.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..utils import (
    affine_transform_gauss_legendre_nodes_and_weights,
    gauss_legendre_nodes_and_weights,
)

__all__ = [
    "HubbleParameter",
    "comoving_distance",
    "matter_density_parameter",
    "redshift_at_comoving_distance",
    "transverse_comoving_distance",
]

type HubbleParameter[P] = Callable[[P, Array], Array]
"""Hubble rate :math:`H(z)` in 1/Mpc as a function of ``(params, z)``."""

_NEWTON_ITERATIONS = 7


def matter_density_parameter[P](
    hubble_parameter: HubbleParameter[P],
    params: P,
    z: Real[ArrayLike, "*shape"],
    omega_m: Real[ArrayLike, ""],
) -> Float[Array, "*shape"]:
    """Matter density parameter :math:`\\Omega_m(z)`.

    Assumes matter that dilutes as :math:`(1 + z)^3` at all redshifts.

    Parameters
    ----------
    hubble_parameter : callable
        ``hubble_parameter(params, z)``, the Hubble rate in 1/Mpc.
    params : pytree
        First argument of ``hubble_parameter``.
    z : ArrayLike
        Redshift, non-negative.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m`.

    Returns
    -------
    Array
        :math:`\\Omega_m (1 + z)^3 H_0^2 / H^2(z)`, dimensionless, with the
        shape of ``z``.
    """
    z = jnp.asarray(z)
    h0 = hubble_parameter(params, jnp.zeros((), z.dtype))
    return jnp.asarray(omega_m) * (1 + z) ** 3 * (h0 / hubble_parameter(params, z)) ** 2


def comoving_distance[P](
    hubble_parameter: HubbleParameter[P],
    params: P,
    z: Real[ArrayLike, "*shape"],
) -> Float[Array, "*shape"]:
    """Line-of-sight comoving distance :math:`\\chi(z)`.

    Parameters
    ----------
    hubble_parameter : callable
        ``hubble_parameter(params, z)``, the Hubble rate in 1/Mpc.
    params : pytree
        First argument of ``hubble_parameter``.
    z : ArrayLike
        Redshift, non-negative.

    Returns
    -------
    Array
        :math:`\\chi(z) = \\int_0^z \\mathrm{d}z' / H(z')` in Mpc, with the
        shape of ``z``.

    Notes
    -----
    A 32-point Gauss–Legendre rule in :math:`\\ln(1 + z)`.
    """
    with jax.ensure_compile_time_eval():
        nodes, weights = gauss_legendre_nodes_and_weights(32)
    length = jnp.log1p(jnp.asarray(z))
    x, weights = affine_transform_gauss_legendre_nodes_and_weights(
        nodes, weights, jnp.zeros_like(length), length
    )
    integrand = jnp.exp(x) / hubble_parameter(params, jnp.expm1(x))
    return jnp.sum(weights * integrand, axis=-1)


def transverse_comoving_distance[P](
    hubble_parameter: HubbleParameter[P],
    params: P,
    z: Real[ArrayLike, "*shape"],
    omega_k: Real[ArrayLike, ""],
) -> Float[Array, "*shape"]:
    """Transverse comoving distance :math:`D_M(z)`.

    With :math:`x = \\Omega_k (H_0 \\chi)^2` this is
    :math:`\\chi\\, \\mathrm{sinhc}(\\sqrt{x})`, using :math:`\\sin` for
    negative :math:`x`. The expression is smooth and differentiable through
    :math:`\\Omega_k = 0`.

    Parameters
    ----------
    hubble_parameter : callable
        ``hubble_parameter(params, z)``, the Hubble rate in 1/Mpc.
    params : pytree
        First argument of ``hubble_parameter``.
    z : ArrayLike
        Redshift, non-negative.
    omega_k : ArrayLike
        Curvature density parameter today, :math:`\\Omega_k`; positive for an
        open universe. The Hubble rate does not determine it.

    Returns
    -------
    Array
        :math:`D_M(z)` in Mpc, with the shape of ``z``.
    """
    chi = comoving_distance(hubble_parameter, params, z)
    h0 = hubble_parameter(params, jnp.zeros((), chi.dtype))
    return chi * _sinhc_sqrt(jnp.asarray(omega_k) * (h0 * chi) ** 2)


def redshift_at_comoving_distance[P](
    hubble_parameter: HubbleParameter[P],
    params: P,
    chi: Real[ArrayLike, "*shape"],
) -> Float[Array, "*shape"]:
    """Inverse of :func:`comoving_distance`.

    Parameters
    ----------
    hubble_parameter : callable
        ``hubble_parameter(params, z)``, the Hubble rate in 1/Mpc.
    params : pytree
        First argument of ``hubble_parameter``.
    chi : ArrayLike
        Comoving distance in Mpc, non-negative and corresponding to a finite
        redshift.

    Returns
    -------
    Array
        Redshift :math:`z(\\chi)`, with the shape of ``chi``.

    Notes
    -----
    A fixed number of Newton iterations in :math:`\\ln(1 + z)`, starting from
    the low-redshift limit :math:`H_0 \\chi`. Derivatives come from the
    implicit function theorem, not from the iterations.
    """
    chi = jnp.asarray(chi, dtype=jnp.result_type(chi, 1.0))

    def residual(x: Array) -> Array:
        return _distance_residual(hubble_parameter, params, chi, x)

    def solve(f: Callable[[Array], Array], initial: Array) -> Array:
        def step(_: int | Array, x: Array) -> Array:
            z = jnp.expm1(x)
            return x - f(x) * hubble_parameter(params, z) / jnp.exp(x)

        return jax.lax.fori_loop(0, _NEWTON_ITERATIONS, step, initial)

    def tangent_solve(g: Callable[[Array], Array], y: Array) -> Array:
        # Each distance has an independent root, so the Jacobian is diagonal.
        return y / g(jnp.ones_like(y))

    h0 = hubble_parameter(params, jnp.zeros((), chi.dtype))
    x = jax.lax.custom_root(residual, chi * h0, solve, tangent_solve)
    return jnp.expm1(x)


def _sinhc_sqrt(x: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    """:math:`\\sinh(\\sqrt{x}) / \\sqrt{x}` for any sign of ``x``.

    Negative ``x`` gives :math:`\\sin(\\sqrt{-x}) / \\sqrt{-x}`. A Taylor series
    covers :math:`|x| < 10^{-2}`, where the closed forms divide by zero; the
    inner ``where`` keeps gradients finite there.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < 1e-2
    safe = jnp.where(small, 1.0, x)
    root = jnp.sqrt(jnp.abs(safe))
    exact = jnp.where(safe > 0, jnp.sinh(root), jnp.sin(root)) / root
    series = 1 + x / 6 + x**2 / 120 + x**3 / 5040
    return jnp.where(small, series, exact)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _distance_residual[P](
    hubble_parameter: HubbleParameter[P],
    params: P,
    chi: Real[ArrayLike, "*shape"],
    x: Real[ArrayLike, "*shape"],
) -> Float[Array, "*shape"]:
    """Distance residual in log-redshift for independent scalar roots."""
    return comoving_distance(hubble_parameter, params, jnp.expm1(x)) - jnp.asarray(chi)


@_distance_residual.defjvp
def _distance_residual_jvp[P](
    hubble_parameter: HubbleParameter[P],
    primals: tuple[P, Array, Array],
    tangents: tuple[P, Array, Array],
) -> tuple[Array, Array]:
    """Use dchi/dx = exp(x)/H(z), retaining parameter derivatives of chi."""
    params, chi, x = primals
    dparams, dchi, dx = tangents
    z = jnp.expm1(x)
    value, dparameter = jax.jvp(
        lambda p: comoving_distance(hubble_parameter, p, z), (params,), (dparams,)
    )
    derivative = dparameter - dchi + jnp.exp(x) / hubble_parameter(params, z) * dx
    return value - chi, derivative
