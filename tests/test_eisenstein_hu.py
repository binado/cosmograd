"""EH98 physical identities and native JAX transformation behaviour."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from cosmograd import EisensteinHu, FlatLCDM, Perturbations
from cosmograd.perturbations import EisensteinHu as ExportedEisensteinHu


@pytest.fixture(
    params=[
        EisensteinHu(FlatLCDM(H0=67.36, omega_m=0.3153), 0.0493, 0.9649, 3.044),
        EisensteinHu(FlatLCDM(H0=72.0, omega_m=0.25), 0.04, 1.0, 3.1),
        EisensteinHu(FlatLCDM(H0=62.0, omega_m=0.4, t_cmb=2.8), 0.07, 0.94, 2.9),
    ],
    ids=["planck", "low-matter", "high-matter"],
)
def eh(request: pytest.FixtureRequest) -> EisensteinHu:
    return request.param


def test_background_and_pytree(eh: EisensteinHu) -> None:
    bg = eh.background
    cosmo = EisensteinHu(bg, eh.omega_b, eh.n_s, eh.ln10as)
    assert cosmo.background is bg
    assert isinstance(cosmo, Perturbations)
    assert ExportedEisensteinHu is EisensteinHu
    np.testing.assert_array_equal(
        jax.tree_util.tree_leaves(cosmo),
        [bg.H0, bg.omega_m, eh.omega_b, eh.n_s, eh.ln10as],
    )
    with pytest.raises(FrozenInstanceError):
        setattr(cosmo, "n_s", 1.0)


def test_primordial_pivot(eh: EisensteinHu) -> None:
    np.testing.assert_allclose(
        eh.primordial_power_spectrum(0.05), 1e-10 * np.exp(eh.ln10as)
    )


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_amplitude_scaling(eh: EisensteinHu, factor: float) -> None:
    scaled = replace(eh, ln10as=eh.ln10as + np.log(factor))
    k, z = jnp.array([0.001, 0.05, 0.2]), jnp.array([0.0, 1.0, 10.0])
    np.testing.assert_allclose(
        scaled.matter_power_spectrum(k, z), factor * eh.matter_power_spectrum(k, z)
    )


def test_growth_normalisation_and_scale_independence(eh: EisensteinHu) -> None:
    k = jnp.array([0.2, 0.001, 0.05])
    z = jnp.array([10.0, 0.0, 1.0])
    np.testing.assert_allclose(eh.growth_function(0.0, k=k), np.ones(3))
    np.testing.assert_allclose(
        eh.growth_function(z[:, None], k=k),
        np.broadcast_to(eh.growth_function(z)[:, None], (3, 3)),
    )
    np.testing.assert_allclose(
        eh.growth_rate(z[:, None], k=k),
        np.broadcast_to(eh.growth_rate(z)[:, None], (3, 3)),
    )
    np.testing.assert_allclose(
        eh.matter_power_spectrum(k, z) / eh.matter_power_spectrum(k, 0.0),
        eh.growth_function(z) ** 2,
    )


@pytest.mark.parametrize("z", [0.0, 0.5, 3.0, 10.0])
def test_growth_rate_derivative(eh: EisensteinHu, z: float) -> None:
    derivative = jax.grad(lambda x: jnp.log(eh.growth_function(x)))(z)
    np.testing.assert_allclose(eh.growth_rate(z), -(1 + z) * derivative, rtol=1e-10)


@pytest.mark.parametrize("z", [0.0, 0.5, 3.0, 10.0])
def test_matter_only_growth(z: float) -> None:
    eh = EisensteinHu(FlatLCDM(H0=70.0, omega_m=1.0), 0.05, 0.96, 3.0)
    np.testing.assert_allclose(eh.growth_function(z), 1 / (1 + z), rtol=1e-10)
    np.testing.assert_allclose(eh.growth_rate(z), 1.0, rtol=1e-10)


@pytest.mark.parametrize(
    "method",
    [
        "growth_function",
        "growth_rate",
        "primordial_power_spectrum",
        "matter_power_spectrum",
    ],
)
def test_jit_and_vmap(eh: EisensteinHu, method: str) -> None:
    # Unsorted inputs also exercise elementwise batching over evaluation points.
    x = jnp.array([0.2, 0.01, 0.07])

    def evaluate(c: EisensteinHu, value: Array) -> Array:
        if method == "matter_power_spectrum":
            return c.matter_power_spectrum(value, 1.0)
        return getattr(c, method)(value)

    expected = evaluate(eh, x)
    np.testing.assert_allclose(jax.jit(evaluate)(eh, x), expected)
    np.testing.assert_allclose(jax.vmap(evaluate, in_axes=(None, 0))(eh, x), expected)
    second = replace(eh, background=replace(eh.background, omega_m=0.33), n_s=0.97)
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), eh, second)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate, in_axes=(0, None)))(batch, x),
        jnp.stack([expected, evaluate(second, x)]),
    )


def test_parameter_gradients(eh: EisensteinHu) -> None:
    gradient = jax.jit(jax.grad(lambda c: jnp.log(c.matter_power_spectrum(0.1, 0.5))))(
        eh
    )
    assert isinstance(gradient.background, FlatLCDM)
    leaves = jnp.array(jax.tree_util.tree_leaves(gradient))
    assert leaves.shape == (5,)
    assert jnp.all(jnp.isfinite(leaves) & (leaves != 0))
    np.testing.assert_allclose(gradient.ln10as, 1.0)


def test_nonlinear_rejected(eh: EisensteinHu) -> None:
    with pytest.raises(NotImplementedError, match="linear power only"):
        replace(eh, nonlinear=True)
