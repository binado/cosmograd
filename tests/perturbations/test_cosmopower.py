"""Physics checks specific to the CosmoPower backend."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import FlatLCDM
from cosmograd.perturbations.cosmopower import CosmoPower
from cosmograd.perturbations.growth import linear_growth, matter_domination_scale_factor

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


@pytest.mark.parametrize("k", [1e-2, 1e-1, 1.0])
def test_growth_is_scale_independent(cosmo: CosmoPower, k: float) -> None:
    z = jnp.linspace(0.0, 4.0, 9)
    np.testing.assert_allclose(
        cosmo.growth_function(z, k=k), cosmo.growth_function(z), rtol=2e-3
    )


@pytest.mark.parametrize("extrapolate", [False, True])
def test_linear_power_continuation(cosmo: CosmoPower, extrapolate: bool) -> None:
    cosmo = replace(cosmo, extrapolate=extrapolate)
    k = jnp.stack((cosmo.k_min / 10, 0.01, 1.0, cosmo.k_max * 10))[:, None]
    z = jnp.array([0.0, 2.0, 5.0, 5.0 - 1e-6])
    legacy = jnp.exp(cosmo._log_pk_extrapolated(k, z))
    np.testing.assert_allclose(cosmo.linear_matter_power_spectrum(k, z), legacy)

    z_high = jnp.array([8.0, 5.0 + 1e-6, 6.0])
    z_mixed = jnp.concatenate((z_high, z))
    power = jax.jit(cosmo.linear_matter_power_spectrum)(k, z_mixed)
    d, _ = linear_growth(
        cosmo.background,
        jnp.append(z_high, 5.0),
        a_initial=matter_domination_scale_factor(cosmo.background),
    )
    anchor = legacy[:, 2:3]
    np.testing.assert_allclose(power[:, :3], anchor * (d[:-1] / d[-1]) ** 2)
    np.testing.assert_allclose(power[:, 3:], legacy)
    np.testing.assert_allclose(power[:, [1, -1]] / anchor, 1.0, rtol=1e-6)

    expected_growth = jnp.sqrt(power / legacy[:, :1])
    np.testing.assert_allclose(cosmo.growth_function(z_mixed, k=k), expected_growth)
    np.testing.assert_allclose(cosmo.matter_power_spectrum(k[:, 0], 8.0), power[:, 0])


def test_growth_rate_at_and_above_boundary(cosmo: CosmoPower) -> None:
    k = jnp.array([0.01, 0.1, 1.0])[:, None]
    z = jnp.array([8.0, 5.0, 6.0, 3.0])
    rate = jax.jit(lambda z: cosmo.growth_rate(z, k=k))(z)
    _, expected_high = linear_growth(
        cosmo.background,
        z[jnp.array([0, 2])],
        a_initial=matter_domination_scale_factor(cosmo.background),
    )
    np.testing.assert_allclose(
        rate[:, [0, 2]], jnp.broadcast_to(expected_high, (3, 2)), rtol=1e-6
    )

    z_emulator = z[jnp.array([1, 3])]
    _, slope = jax.jvp(
        lambda z: cosmo._log_pk_extrapolated(k, z),
        (z_emulator,),
        (jnp.ones_like(z_emulator),),
    )
    np.testing.assert_allclose(rate[:, [1, 3]], -0.5 * (1 + z_emulator) * slope)


def test_redshifts_before_ode_initialization(cosmo: CosmoPower) -> None:
    z_initial = 1 / matter_domination_scale_factor(cosmo.background) - 1
    z = jnp.array([8.0, 2 * z_initial, 3.0])
    results = (
        cosmo.linear_matter_power_spectrum(0.01, z),
        cosmo.growth_function(z),
        cosmo.growth_rate(z),
    )
    for result in results:
        assert jnp.isnan(result[1])
        assert jnp.all(jnp.isfinite(result[jnp.array([0, 2])]))


def test_mixed_redshift_cosmological_derivatives(cosmo: CosmoPower) -> None:
    z = jnp.array([8.0, 3.0, 5.0])

    def log_power(omega_m: float) -> jax.Array:
        c = replace(cosmo, background=replace(cosmo.background, omega_m=omega_m))
        return jnp.log(c.linear_matter_power_spectrum(0.01, z))

    omega_m = 0.315
    forward = jax.jit(jax.jacfwd(log_power))(omega_m)
    reverse = jax.jit(jax.jacrev(log_power))(omega_m)
    assert jnp.all(jnp.isfinite(forward))
    np.testing.assert_allclose(forward, reverse, rtol=1e-6)

    compiled = jax.jit(log_power)
    step = 1e-4
    delta = compiled(omega_m + step) - compiled(omega_m - step)
    ratio_derivative = (delta[0] - delta[2]) / (2 * step)
    assert abs(ratio_derivative) > 1e-4
    np.testing.assert_allclose(forward[0] - forward[2], ratio_derivative, rtol=1e-3)


def test_nonlinear_high_redshift_keeps_legacy_prediction(cosmo: CosmoPower) -> None:
    nonlinear = replace(cosmo, nonlinear=True)
    k, z = jnp.asarray(0.1), jnp.array([3.0, 5.0, 8.0])
    legacy = jnp.exp(
        cosmo._log_pk_extrapolated(k, z) + cosmo._interpolate("mpk_boost", k, z)
    )
    np.testing.assert_allclose(jax.jit(nonlinear.matter_power_spectrum)(k, z), legacy)
    np.testing.assert_allclose(
        nonlinear.linear_matter_power_spectrum(k, z),
        cosmo.linear_matter_power_spectrum(k, z),
    )
    np.testing.assert_allclose(nonlinear.growth_function(z), cosmo.growth_function(z))
    np.testing.assert_allclose(nonlinear.growth_rate(z), cosmo.growth_rate(z))
