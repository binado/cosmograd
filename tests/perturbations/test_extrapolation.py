"""Extrapolation of the linear spectrum outside the trusted k range."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd.perturbations._extrapolation import extrapolate_log_pk
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

from ..cosmologies import EMULATOR_COSMOLOGY
from ..optional import requires

K_MIN, K_MAX = 1e-3, 1.0
N_S = 0.96


def primordial(k):
    return (k / 0.05) ** (N_S - 1)


def power_law(k, z):
    return 2.0 + 1.5 * jnp.log(k) - 0.3 * jnp.log1p(z)


def curved(k, z):
    return power_law(k, z) - 0.2 * jnp.log(k) ** 2


@pytest.fixture
def k_low():
    return jnp.geomspace(1e-6, K_MIN, 5)


@pytest.fixture
def k_high():
    return jnp.geomspace(K_MAX, 1e2, 5)


def extrapolate(log_pk, k, z=0.5, slope_fraction=0.9):
    return extrapolate_log_pk(log_pk, primordial, k, z, K_MIN, K_MAX, slope_fraction)


def test_power_law_reproduced_above_kmax(k_high):
    np.testing.assert_allclose(
        extrapolate(power_law, k_high), power_law(k_high, 0.5), rtol=1e-10
    )


def test_low_k_tail_scales_as_k_primordial(k_low):
    out = extrapolate(power_law, k_low)
    expected = power_law(K_MIN, 0.5) + jnp.log(
        k_low * primordial(k_low) / (K_MIN * primordial(K_MIN))
    )
    np.testing.assert_allclose(out, expected, rtol=1e-10)


@pytest.mark.parametrize("edge", [K_MIN, K_MAX])
def test_continuous_at_edges(edge):
    k = jnp.array([edge * (1 - 1e-9), edge, edge * (1 + 1e-9)])
    out = extrapolate(curved, k)
    np.testing.assert_allclose(out, out[1], rtol=1e-6)


def test_slope_fraction_matters_only_for_curved_input(k_high):
    k = k_high[1:]
    np.testing.assert_allclose(
        extrapolate(power_law, k, slope_fraction=0.5),
        extrapolate(power_law, k, slope_fraction=0.9),
        rtol=1e-10,
    )
    assert not np.allclose(
        extrapolate(curved, k, slope_fraction=0.5),
        extrapolate(curved, k, slope_fraction=0.9),
    )


def test_gradient_finite_at_every_branch(k_low, k_high):
    k = jnp.concatenate([k_low, k_high, jnp.array([0.1])])
    grad = jax.grad(lambda a: extrapolate(lambda q, z: a * curved(q, z), k).sum())(1.0)
    assert np.isfinite(grad)


@pytest.fixture(
    params=[
        pytest.param(
            CosmoPower(EMULATOR_COSMOLOGY),
            id="cosmopower",
            marks=requires("cosmopower_jax"),
        ),
        pytest.param(
            BaccoEmu(EMULATOR_COSMOLOGY),
            id="baccoemu",
            marks=requires("baccoemu.matter_powerspectrum"),
        ),
    ]
)
def emulator(request):
    return request.param


Z = jnp.array([0.0, 0.5])


def test_k_range_is_ordered(emulator):
    assert emulator.k_min < emulator.k_max


def test_in_range_unchanged(emulator):
    k = jnp.geomspace(emulator.k_min * 1.01, emulator.k_max * 0.99, 9)[:, None]
    on = replace(emulator, extrapolate=True)
    np.testing.assert_allclose(
        on.linear_matter_power_spectrum(k, Z),
        emulator.linear_matter_power_spectrum(k, Z),
        rtol=1e-6,
    )


def test_out_of_range_tails(emulator):
    on = replace(emulator, extrapolate=True)
    k_lo = emulator.k_min * jnp.geomspace(1e-3, 1.0, 5)[:, None]
    k_hi = emulator.k_max * jnp.geomspace(1.0, 10.0, 5)[:, None]
    low = on.linear_matter_power_spectrum(k_lo, Z)
    high = on.linear_matter_power_spectrum(k_hi, Z)
    assert np.all(np.isfinite(low)) and np.all(low > 0)
    assert np.all(np.isfinite(high)) and np.all(high > 0)
    assert np.all(np.diff(high, axis=0) < 0)
    # Constant transfer function: P / (k P_R) is constant at low k.
    ratio = low / (k_lo * on.primordial_power_spectrum(k_lo))
    np.testing.assert_allclose(
        ratio, np.broadcast_to(ratio[-1], ratio.shape), rtol=1e-6
    )


def test_grad_jit_vmap_finite(emulator):
    k = jnp.geomspace(emulator.k_min * 1e-2, emulator.k_max * 10, 7)

    def total(h):
        cosmo = replace(
            emulator,
            background=replace(emulator.background, H0=h),
            extrapolate=True,
        )
        return jnp.log(cosmo.linear_matter_power_spectrum(k, 0.3)).sum()

    h0 = emulator.background.H0
    grad = jax.jit(jax.grad(total))(h0)
    assert np.isfinite(grad)
    batched = jax.vmap(total)(h0 * jnp.array([0.98, 1.0, 1.02]))
    assert np.all(np.isfinite(batched))


def test_growth_matches_spectrum_out_of_range(emulator):
    on = replace(emulator, extrapolate=True)
    k = jnp.array([emulator.k_min * 1e-2, emulator.k_max * 5])[:, None]
    expected = jnp.sqrt(
        on.linear_matter_power_spectrum(k, Z) / on.linear_matter_power_spectrum(k, 0.0)
    )
    np.testing.assert_allclose(on.growth_function(Z, k=k), expected, rtol=1e-5)
