"""Checks for quadrature utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd.utils import (
    affine_transform_gauss_legendre_nodes,
    gauss_legendre_nodes_and_weights,
)


@pytest.mark.parametrize("n", [1, 2, 8, 32, 64, 128])
def test_gauss_legendre_polynomial_exactness(n: int) -> None:
    nodes, weights = jax.jit(gauss_legendre_nodes_and_weights, static_argnums=0)(n)
    transformed = affine_transform_gauss_legendre_nodes(nodes, 0.0, 1.0)
    # The highest polynomial degree integrated exactly by an n-point rule.
    integral = jnp.sum(weights * transformed ** (2 * n - 1)) / 2
    np.testing.assert_allclose(integral, 1 / (2 * n))


@pytest.mark.parametrize("upper", [2.0, jnp.array([2.0, 4.0])])
def test_affine_transform_integral_and_gradient(upper: jax.Array | float) -> None:
    nodes, weights = gauss_legendre_nodes_and_weights(8)

    def integral(b: jax.Array | float) -> jax.Array:
        a = jnp.zeros_like(b)
        x = affine_transform_gauss_legendre_nodes(nodes, a, b)
        return b / 2 * jnp.sum(weights * x**2, axis=-1)

    np.testing.assert_allclose(jax.jit(integral)(upper), jnp.asarray(upper) ** 3 / 3)
    gradient = jax.grad(lambda b: jnp.sum(integral(b)))(upper)
    np.testing.assert_allclose(gradient, jnp.asarray(upper) ** 2)
