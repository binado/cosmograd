"""Numerical utilities shared by cosmology implementations."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

__all__ = [
    "affine_transform_gauss_legendre_nodes_and_weights",
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

    Notes
    -----
    Computes nodes and weights with NumPy at trace time, then converts them to
    JAX arrays.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


def affine_transform_gauss_legendre_nodes_and_weights(
    nodes: Real[ArrayLike, " n"],
    weights: Real[ArrayLike, " n"],
    a: Real[ArrayLike, "*shape"],
    b: Real[ArrayLike, "*shape"],
) -> tuple[Float[Array, "*shape n"], Float[Array, "*shape n"]]:
    """Map Gauss–Legendre nodes and weights from ``[-1, 1]`` to ``[a, b]``.

    Parameters
    ----------
    nodes, weights : ArrayLike
        Nodes and corresponding weights on ``[-1, 1]``, each of shape ``(n,)``.
    a, b : ArrayLike
        Integration bounds with matching shapes. Scalars are supported.

    Returns
    -------
    nodes, weights : Array
        Transformed nodes and weights, each with a final quadrature axis.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    scale = (b - a)[..., None] / 2
    transformed_nodes = a[..., None] + scale * (jnp.asarray(nodes) + 1)
    transformed_weights = scale * jnp.asarray(weights)
    return transformed_nodes, transformed_weights
