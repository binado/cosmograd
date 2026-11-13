"""Checks for quadrature utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd.utils import (
    affine_transform_gauss_legendre_nodes,
    gauss_legendre_nodes_and_weights,
)


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
