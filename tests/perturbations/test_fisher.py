"""End to end: a Fisher matrix of the power spectrum from a jitted Jacobian."""

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from .conftest import PerturbationsCase

pytestmark = pytest.mark.slow


def test_fisher_matrix_of_power_spectrum_is_finite_with_positive_diagonal(
    perturbations_case: PerturbationsCase,
) -> None:
    """Every background parameter moves the spectrum, and no gradient is NaN.

    A parameter the pipeline silently disconnects, through a clip or a static
    field, shows up as a zero on the diagonal. The covariance is the identity
    since only the Jacobian is under test.
    """
    from dataclasses import replace

    cosmo, k, z = perturbations_case.cosmo, perturbations_case.k, perturbations_case.z
    parameters, unravel = ravel_pytree(cosmo.background)

    def log_power(vector: jax.Array) -> jax.Array:
        varied = replace(cosmo, background=unravel(vector))
        return jnp.log(varied.matter_power_spectrum(k[:, None], z[None, :])).ravel()

    jacobian = jax.jit(jax.jacfwd(log_power))(parameters)
    fisher = jacobian.T @ jacobian
    assert jnp.all(jnp.isfinite(fisher))
    assert jnp.all(jnp.diag(fisher) > 0)
