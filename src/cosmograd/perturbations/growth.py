"""Scale-independent linear growth on a supplied expansion history."""

import diffrax
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..background import Background

__all__ = ["linear_growth"]


def _growth_rhs(x: Real[ArrayLike, ""], y: Array, background: Background) -> Array:
    """Evolve (D/a, a^(1/2) (H/H0) dD/d ln a) without differentiating H."""
    x = jnp.asarray(x)
    z = jnp.expm1(-x)
    a32_e = (
        jnp.exp(1.5 * x) * background.hubble_parameter(z) / background.hubble_constant
    )
    omega_m = background.matter_density_parameter(z)
    g, u = y
    return jnp.stack((u / a32_e - g, 1.5 * (a32_e * omega_m * g - u)))


def linear_growth(
    background: Background,
    z: Real[ArrayLike, "*shape"],
    *,
    normalize: bool = True,
    a_initial: float = 1e-3,
    rtol: float | None = None,
    atol: float | None = None,
    max_steps: int = 512,
) -> tuple[Float[Array, "*shape"], Float[Array, "*shape"]]:
    r"""Solve for linear growth D and logarithmic growth rate f with Diffrax.

    Parameters
    ----------
    background : Background
        Expansion history and matter density. Must be a JAX pytree to
        differentiate or batch over cosmological parameters.
    z : ArrayLike
        Redshifts in [0, 1/a_initial - 1], in any order and shape.
    normalize : bool, default True
        Return D(z)/D(0). If False, retain the matter-era normalization D = a
        at the initial scale factor. Does not affect f. Static.
    a_initial : float, default 0.001
        Initial scale factor, in (0, 1). Must lie in matter domination, with
        negligible radiation, curvature and dark energy. Initial conditions
        are D = a and dD/d ln a = a. Static under JAX transformations.
    rtol, atol : float, optional
        Relative and absolute tolerances for the state
        (D/a, a^(1/2) (H/H_0) dD/d ln a). Defaults are max(1e-8, 10 eps) and
        max(1e-10, 10 eps), respectively, where eps is machine precision in
        the active JAX floating dtype.
    max_steps : int, default 512
        Maximum integration steps and dense-output storage. Static.

    Returns
    -------
    D, f : Array
        Growth and f = d ln D / d ln a, each with the shape of z.

    Notes
    -----
    For pressureless clustering matter in GR, with other components smooth
    on the scales of interest, the subhorizon growth equation is

    .. math::

        D'' + \bigl(2 + (\ln H)'\bigr)D' - \tfrac32\Omega_m(a)D = 0,

    with primes denoting derivatives with respect to :math:`x = \ln a`.
    Multiplying by :math:`a^2 E(a)`, where :math:`E(a) = H(a)/H_0`, puts the
    equation in Sturm–Liouville form without derivatives of :math:`H`:

    .. math::

        \bigl(a^2 E\, D'\bigr)' = \tfrac32 a^2 E\,\Omega_m(a)D.

    To factor out the matter-era behaviour :math:`D \propto a` and
    :math:`E \propto a^{-3/2}`, the solver evolves the dimensionless state

    .. math::

        g = \frac{D}{a}, \qquad u = a^{1/2} E\, D' = a^{3/2} E\,(g + g').

    Substituting :math:`a^2 E\, D' = a^{3/2} u` into the Sturm–Liouville
    equation and using :math:`(a^{3/2} u)' = a^{3/2}(\tfrac32 u + u')` gives
    the first-order system

    .. math::

        g' &= \frac{u}{a^{3/2} E(a)} - g, \\
        u' &= \tfrac32\bigl(a^{3/2} E(a)\Omega_m(a)g - u\bigr),

    with matter-era initial conditions :math:`D(a_i) = a_i` and
    :math:`D'(a_i) = a_i`,

    .. math::

        g(a_i) = 1, \qquad u(a_i) = a_i^{3/2} E(a_i),

    from which the growth factor and logarithmic growth rate are recovered as

    .. math::

        D(a) = a\,g(a), \qquad
        f(a) = \frac{D'}{D} = \frac{u(a)}{a^{3/2} E(a)\,g(a)}.

    This supports curvature and smooth dark energy through the supplied
    background; it does not model scale-dependent growth from massive
    neutrinos or modified gravity.

    Radiation in the supplied H(a) is included, but the matter-era initial
    conditions are inappropriate during radiation domination. Choose
    a_initial within matter domination and check sensitivity to that choice;
    solver tolerances do not control the initial-condition approximation.

    A single Tsit5 solve runs from ln(a_initial) to today. Dense output handles
    unsorted, repeated and multidimensional redshifts without extra solves.
    DirectAdjoint supports both forward- and reverse-mode differentiation.
    No inputs are validated, and out-of-range redshifts are unsupported.
    """
    z = jnp.asarray(z, dtype=float)
    eps = float(jnp.finfo(z.dtype).eps)
    rtol = max(1e-8, 10 * eps) if rtol is None else rtol
    atol = max(1e-10, 10 * eps) if atol is None else atol
    u0 = (
        a_initial**1.5
        * background.hubble_parameter(1 / a_initial - 1)
        / background.hubble_constant
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_growth_rhs),
        diffrax.Tsit5(),
        t0=jnp.log(a_initial),
        t1=0.0,
        dt0=0.01,
        y0=jnp.stack((jnp.ones_like(u0), u0)),
        args=background,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(dense=True),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=max_steps,
    )
    x = -jnp.log1p(z)
    values = jax.vmap(solution.evaluate)(x.reshape(-1)).reshape(*z.shape, 2)
    g, u = values[..., 0], values[..., 1]
    d = g / (1 + z)
    if normalize:
        d = d / solution.evaluate(0.0)[0]
    a32_e = (
        (1 + z) ** -1.5 * background.hubble_parameter(z) / background.hubble_constant
    )
    return d, u / (a32_e * g)
