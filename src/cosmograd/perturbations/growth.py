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
    """Evolve (D/a, d(D/a)/d ln a) to avoid small early-time amplitudes."""
    x = jnp.asarray(x)
    z = jnp.expm1(-x)
    dlnh_dx = jax.grad(lambda x_: jnp.log(background.hubble_parameter(jnp.expm1(-x_))))(
        x
    )
    omega_m = background.matter_density_parameter(z)
    g, v = y
    return jnp.stack((v, -(4 + dlnh_dx) * v - (3 + dlnh_dx - 1.5 * omega_m) * g))


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
        Differentiable expansion history and matter density. Must be a JAX
        pytree to differentiate or batch over cosmological parameters.
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
        Relative and absolute tolerances for the state (D/a, d(D/a)/d ln a).
        Defaults are max(1e-8, 10 eps) and max(1e-10, 10 eps), respectively,
        where eps is machine precision in the active JAX floating dtype.
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

        D'' + (2 + (\ln H)')D' - \tfrac32\Omega_m(a)D = 0,

    with primes denoting derivatives with respect to ln a. This supports
    curvature and smooth dark energy through the supplied background; it
    does not model scale-dependent growth from massive neutrinos or modified
    gravity. The background must be differentiable with respect to redshift.

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
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_growth_rhs),
        diffrax.Tsit5(),
        t0=jnp.log(a_initial),
        t1=0.0,
        dt0=0.01,
        y0=jnp.array([1.0, 0.0]),
        args=background,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(dense=True),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=max_steps,
    )
    x = -jnp.log1p(z)
    values = jax.vmap(solution.evaluate)(x.reshape(-1)).reshape(*z.shape, 2)
    g, v = values[..., 0], values[..., 1]
    d = g / (1 + z)
    if normalize:
        d = d / solution.evaluate(0.0)[0]
    return d, 1 + v / g
