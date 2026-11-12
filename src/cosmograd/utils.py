"""Numerical utilities shared by cosmology implementations."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

__all__ = [
    "affine_transform_gauss_legendre_nodes",
    "gauss_legendre_nodes_and_weights",
]


def gauss_legendre_nodes_and_weights(
    n: int,
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Return Gauss–Legendre quadrature nodes and weights on ``[-1, 1]``.

    Parameters
    ----------
    n : int
        Positive quadrature order. Must be static under JAX transformations.

    Returns
    -------
    nodes, weights : Array
        Increasing nodes and corresponding weights, each of shape ``(n,)``.
        Compute once and reuse across integrations. Precision follows the JAX
        configuration.

    Notes
    -----
    Computes nodes and weights with NumPy at trace time, then converts them to
    JAX arrays.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


def affine_transform_gauss_legendre_nodes(
    nodes: Real[ArrayLike, " n"],
    a: Real[ArrayLike, "*shape"],
    b: Real[ArrayLike, "*shape"],
) -> Float[Array, "*shape n"]:
    """Map quadrature nodes from ``[-1, 1]`` to ``[a, b]``.

    Parameters
    ----------
    nodes : ArrayLike
        Nodes on ``[-1, 1]``, shape ``(n,)``.
    a, b : ArrayLike
        Integration bounds with matching shapes. Scalars are supported.

    Returns
    -------
    Array
        Transformed nodes, with a final quadrature axis. Multiply the weighted
        sum of integrand values by ``(b - a) / 2`` to obtain the integral.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    return a[..., None] + (b - a)[..., None] * (jnp.asarray(nodes) + 1) / 2
