"""Exact relations every background satisfies, whatever the cosmology."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import W0WaCDM

from ..cosmologies import Cosmology, cosmology_params

Z = jnp.linspace(0.0, 5.0, 50)


@pytest.fixture(params=cosmology_params())
def cosmo(request: pytest.FixtureRequest) -> Cosmology:
    return request.param


def test_redshift_at_comoving_distance_inverts_comoving_distance(
    cosmo: Cosmology,
) -> None:
    chi = cosmo.comoving_distance(Z)
    np.testing.assert_allclose(
        cosmo.redshift_at_comoving_distance(chi), Z, rtol=1e-6, atol=1e-6
    )


def test_luminosity_distance_is_etherington_relation_of_angular_diameter_distance(
    cosmo: Cosmology,
) -> None:
    np.testing.assert_allclose(
        cosmo.luminosity_distance(Z),
        (1 + Z) ** 2 * cosmo.angular_diameter_distance(Z),
        rtol=1e-12,
    )


@pytest.mark.parametrize("z", [0.0, 1.0, 1200.0])
def test_inverse_distance_derivatives_follow_hubble_parameter(
    cosmo: Cosmology, z: float
) -> None:
    chi = cosmo.comoving_distance(z)
    inverse = cosmo.redshift_at_comoving_distance
    first = jax.jit(jax.grad(inverse))(chi)
    second = jax.jit(jax.grad(jax.grad(inverse)))(chi)
    hubble = cosmo.hubble_parameter(z)
    np.testing.assert_allclose(first, hubble)
    np.testing.assert_allclose(second, jax.grad(cosmo.hubble_parameter)(z) * hubble)


def test_comoving_distance_derivative_is_inverse_hubble_parameter(
    cosmo: Cosmology,
) -> None:
    z = jnp.array([0.0, 1e-8, 0.1, 1.0, 100.0, 1200.0])
    derivative = jax.jit(jax.vmap(jax.grad(cosmo.comoving_distance)))(z)
    np.testing.assert_allclose(derivative, 1 / cosmo.hubble_parameter(z))


@pytest.mark.parametrize("z", [0.5, 1200.0])
def test_inverse_distance_forward_and_reverse_gradients_agree(
    cosmo: Cosmology, z: float
) -> None:
    chi = cosmo.comoving_distance(z)
    reverse = jax.jit(jax.grad(lambda c: c.redshift_at_comoving_distance(chi)))(cosmo)
    forward = jax.jit(jax.jacfwd(lambda c: c.redshift_at_comoving_distance(chi)))(cosmo)
    np.testing.assert_allclose(jax.tree.leaves(forward), jax.tree.leaves(reverse))


@pytest.mark.parametrize("omega_k", [0.0, 1e-8, -1e-8, 1e-3])
def test_curvature_gradient_is_finite_near_flat(omega_k: float) -> None:
    cosmo = W0WaCDM(H0=67.36, omega_m=0.3153, omega_k=omega_k)
    grad = jax.grad(lambda c: c.luminosity_distance(1.0))(cosmo)
    assert np.isfinite(grad.omega_k)
