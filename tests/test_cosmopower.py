"""Physics checks specific to the CosmoPower backend."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import FlatLCDM
from cosmograd.constants import C_KM_S
from cosmograd.perturbations.cosmopower import K_PIVOT, CosmoPower

pytest.importorskip("cosmopower_jax")


@pytest.fixture
def cosmo() -> CosmoPower:
    return CosmoPower(
        FlatLCDM(H0=67.4, omega_m=0.315, omega_b=0.0493, n_s=0.965, ln10as=3.044)
    )


def test_background_and_pytree(cosmo: CosmoPower) -> None:
    bg = replace(cosmo.background, t_cmb=2.8)
    cosmo = replace(cosmo, background=bg)
    assert cosmo.background is bg
    np.testing.assert_array_equal(
        jax.tree_util.tree_leaves(cosmo),
        [
            bg.H0,
            bg.omega_m,
            bg.omega_b,
            bg.n_s,
            bg.ln10as,
            cosmo.c_min,
            cosmo.eta_0,
        ],
    )


def test_background_parameter_gradients(cosmo: CosmoPower) -> None:
    gradient = jax.jit(jax.grad(lambda c: jnp.log(c.matter_power_spectrum(0.1, 0.5))))(
        cosmo
    )
    assert isinstance(gradient.background, FlatLCDM)
    derivatives = jnp.array([gradient.background.H0, gradient.background.omega_m])
    assert jnp.all(jnp.isfinite(derivatives) & (derivatives != 0))


def test_sigma8_planck(cosmo: CosmoPower) -> None:
    """Planck-like parameters with massless neutrinos give sigma_8 near 0.82."""
    k = jnp.geomspace(1e-4, 5.0, 4000)
    kr = k * 8 / (cosmo.background.hubble_constant * C_KM_S / 100)
    window = 3 * (jnp.sin(kr) - kr * jnp.cos(kr)) / kr**3
    integrand = k**3 * cosmo.matter_power_spectrum(k, 0.0) * window**2
    sigma8 = jnp.sqrt(jnp.trapezoid(integrand, jnp.log(k)) / (2 * np.pi**2))
    np.testing.assert_allclose(sigma8, 0.82, rtol=0.02)


def test_primordial_amplitude_at_pivot(cosmo: CosmoPower) -> None:
    np.testing.assert_allclose(
        jnp.log(1e10 * cosmo.primordial_power_spectrum(K_PIVOT)),
        np.asarray(cosmo.background.ln10as),
    )


@pytest.mark.parametrize("k", [1e-3, 1e-2, 1e-1, 1.0])
def test_power_spectrum_gradient_in_amplitude(cosmo: CosmoPower, k: float) -> None:
    """The linear spectrum is proportional to A_s, so d ln P / d ln(1e10 A_s) = 1."""
    grad = jax.grad(lambda c: jnp.log(c.matter_power_spectrum(k, 1.0)))(cosmo)
    np.testing.assert_allclose(grad.background.ln10as, 1.0, rtol=1e-2)


@pytest.mark.parametrize("k", [1e-2, 1e-1, 1.0])
def test_growth_is_scale_independent(cosmo: CosmoPower, k: float) -> None:
    z = jnp.linspace(0.0, 4.0, 9)
    np.testing.assert_allclose(
        cosmo.growth_function(z, k=k), cosmo.growth_function(z), rtol=2e-3
    )


def test_growth_rate_approximation(cosmo: CosmoPower) -> None:
    """Linder (2005): f is close to Omega_m(z)^0.55 in LambdaCDM."""
    z = jnp.linspace(0.0, 4.0, 9)
    omega_m_z = cosmo.background.matter_density_parameter(z)
    np.testing.assert_allclose(cosmo.growth_rate(z), omega_m_z**0.55, rtol=1e-2)
