from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import FlatLCDM

H0, OMEGA_M, T_CMB, N_EFF = 67.36, 0.3153, 2.7255, 3.044
RTOL = 1e-5


@pytest.fixture(scope="module")
def cosmo() -> FlatLCDM:
    return FlatLCDM(H0=H0, omega_m=OMEGA_M, t_cmb=T_CMB, n_eff=N_EFF)


@pytest.mark.parametrize(("H0", "omega_m"), [(55.0, 0.2), (67.36, 0.3153), (85.0, 0.4)])
def test_inverse_round_trip(H0: float, omega_m: float) -> None:
    cosmo = FlatLCDM(H0=H0, omega_m=omega_m)
    z = jnp.array([[0.0, 1e-8, 0.1], [1.0, 100.0, 1200.0]])
    inverse = jax.jit(cosmo.redshift_at_comoving_distance)
    np.testing.assert_allclose(inverse(cosmo.comoving_distance(z)), z)


@pytest.mark.parametrize("z", [0.0, 1.0, 1200.0])
def test_inverse_distance_derivatives(cosmo: FlatLCDM, z: float) -> None:
    chi = cosmo.comoving_distance(z)
    inverse = cosmo.redshift_at_comoving_distance
    first = jax.jit(jax.grad(inverse))(chi)
    second = jax.jit(jax.grad(jax.grad(inverse)))(chi)
    hubble = cosmo.hubble_parameter(z)
    np.testing.assert_allclose(first, hubble)
    np.testing.assert_allclose(second, jax.grad(cosmo.hubble_parameter)(z) * hubble)


@pytest.mark.parametrize("name", ["H0", "omega_m"])
@pytest.mark.parametrize("z", [0.0, 1.0, 1200.0])
def test_inverse_parameter_gradient(cosmo: FlatLCDM, name: str, z: float) -> None:
    chi = cosmo.comoving_distance(z)
    reverse = jax.jit(jax.grad(lambda c: c.redshift_at_comoving_distance(chi)))(cosmo)
    forward = jax.jit(jax.jacfwd(lambda c: c.redshift_at_comoving_distance(chi)))(cosmo)
    eps = 1e-6
    up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
    down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
    finite_difference = (
        up.redshift_at_comoving_distance(chi) - down.redshift_at_comoving_distance(chi)
    ) / (2 * eps)
    np.testing.assert_allclose(getattr(reverse, name), finite_difference, rtol=RTOL)
    np.testing.assert_allclose(getattr(forward, name), getattr(reverse, name))


def test_leaves_are_parameters(cosmo: FlatLCDM) -> None:
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves == [H0, OMEGA_M]


@pytest.mark.parametrize("name", ["H0", "omega_m"])
def test_gradient_matches_finite_difference(cosmo: FlatLCDM, name: str) -> None:
    z = 1.0
    grad = jax.grad(lambda c: c.luminosity_distance(z))(cosmo)
    eps = 1e-6
    up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
    down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
    fd = (up.luminosity_distance(z) - down.luminosity_distance(z)) / (2 * eps)
    np.testing.assert_allclose(getattr(grad, name), fd, rtol=1e-5)


def test_distance_redshift_derivative(cosmo: FlatLCDM) -> None:
    z = jnp.array([0.0, 1e-8, 0.1, 1.0, 100.0, 1200.0])
    derivative = jax.jit(jax.vmap(jax.grad(cosmo.comoving_distance)))(z)
    np.testing.assert_allclose(derivative, 1 / cosmo.hubble_parameter(z))


def test_vmap_over_cosmologies() -> None:
    batch = FlatLCDM(H0=jnp.array([60.0, 70.0]), omega_m=jnp.array([0.3, 0.3]))
    chi = jax.vmap(lambda c: c.comoving_distance(1.0))(batch)
    assert chi.shape == (2,)
    # Larger H0 means a shorter distance in Mpc.
    assert chi[0] > chi[1]
    z = jax.jit(jax.vmap(lambda c, d: c.redshift_at_comoving_distance(d)))(batch, chi)
    np.testing.assert_allclose(z, jnp.ones(2))
