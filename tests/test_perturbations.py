"""Protocol-level checks run against every perturbations backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from cosmograd import Background, Perturbations

SHAPES = [
    pytest.param(0.1, 1.0, (), id="scalars"),
    pytest.param(jnp.geomspace(1e-3, 1.0, 7), 0.5, (7,), id="k-vector"),
    pytest.param(0.1, jnp.linspace(0.0, 3.0, 4), (4,), id="z-vector"),
    pytest.param(
        jnp.geomspace(1e-3, 1.0, 7)[:, None],
        jnp.linspace(0.0, 3.0, 4)[None, :],
        (7, 4),
        id="outer",
    ),
]


def test_isinstance(perturbations: Perturbations) -> None:
    assert isinstance(perturbations, Perturbations)
    assert isinstance(perturbations.background, Background)


def test_leaves_are_floating(perturbations: Perturbations) -> None:
    leaves = jax.tree_util.tree_leaves(perturbations)
    assert leaves
    for leaf in leaves:
        assert jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize(("k", "z", "shape"), SHAPES)
def test_power_spectrum_shape(
    perturbations: Perturbations,
    k: Array,
    z: Array,
    shape: tuple[int, ...],
    nonlinear: bool,
) -> None:
    pk = perturbations.matter_power_spectrum(k, z, nonlinear=nonlinear)
    assert pk.shape == shape
    assert jnp.all(jnp.isfinite(pk)) and jnp.all(pk > 0)


@pytest.mark.parametrize("method", ["growth_function", "growth_rate"])
@pytest.mark.parametrize(("k", "z", "shape"), SHAPES)
def test_growth_shape(
    perturbations: Perturbations,
    method: str,
    k: Array,
    z: Array,
    shape: tuple[int, ...],
) -> None:
    assert getattr(perturbations, method)(z, k=k).shape == shape
    assert getattr(perturbations, method)(z).shape == jnp.shape(z)


def test_outer_grid_matches_elementwise(perturbations: Perturbations) -> None:
    k = jnp.geomspace(1e-3, 1.0, 5)
    z = jnp.linspace(0.0, 2.0, 3)
    kk, zz = jnp.meshgrid(k, z, indexing="ij")
    np.testing.assert_allclose(
        perturbations.matter_power_spectrum(k[:, None], z[None, :]),
        perturbations.matter_power_spectrum(kk.ravel(), zz.ravel()).reshape(kk.shape),
        rtol=1e-12,
    )


def test_growth_normalised_today(perturbations: Perturbations) -> None:
    np.testing.assert_allclose(perturbations.growth_function(0.0), 1.0)


def test_growth_decreases_with_redshift(perturbations: Perturbations) -> None:
    growth = perturbations.growth_function(jnp.linspace(0.0, 3.0, 20))
    assert jnp.all(jnp.diff(growth) < 0)


def test_growth_rate_is_log_derivative(perturbations: Perturbations) -> None:
    z = jnp.linspace(0.0, 3.0, 301)
    ln_d = jnp.log(perturbations.growth_function(z))
    numerical = -(1 + z) * np.gradient(ln_d, z)
    np.testing.assert_allclose(
        perturbations.growth_rate(z)[1:-1], numerical[1:-1], rtol=1e-3
    )


def test_growth_matches_spectrum_ratio(perturbations: Perturbations) -> None:
    k, z = 0.1, jnp.array([0.5, 1.0, 2.0])
    ratio = perturbations.matter_power_spectrum(k, z) / (
        perturbations.matter_power_spectrum(k, 0.0)
    )
    np.testing.assert_allclose(
        perturbations.growth_function(z, k=k) ** 2, ratio, rtol=1e-3
    )


def test_nonlinear_matches_linear_on_large_scales(
    perturbations: Perturbations,
) -> None:
    k, z = jnp.geomspace(1e-4, 1e-3, 5), 0.5
    np.testing.assert_allclose(
        perturbations.matter_power_spectrum(k, z, nonlinear=True),
        perturbations.matter_power_spectrum(k, z),
        rtol=1e-2,
    )


def test_primordial_shape(perturbations: Perturbations) -> None:
    k = jnp.geomspace(1e-4, 1.0, 6).reshape(2, 3)
    result = perturbations.primordial_power_spectrum(k)
    assert result.shape == k.shape
    assert jnp.all(result > 0)


def test_jit_matches_eager(perturbations: Perturbations) -> None:
    k, z = jnp.geomspace(1e-3, 1.0, 5), jnp.linspace(0.0, 2.0, 5)
    np.testing.assert_allclose(
        jax.jit(lambda c: c.matter_power_spectrum(k, z, nonlinear=True))(perturbations),
        perturbations.matter_power_spectrum(k, z, nonlinear=True),
        rtol=1e-10,
    )


@pytest.mark.parametrize("nonlinear", [False, True])
def test_jacobian_is_finite(perturbations: Perturbations, nonlinear: bool) -> None:
    k, z = jnp.geomspace(1e-3, 1.0, 5), 0.5
    jac = jax.jacfwd(lambda c: c.matter_power_spectrum(k, z, nonlinear=nonlinear))(
        perturbations
    )
    assert jax.tree_util.tree_structure(jac) == jax.tree_util.tree_structure(
        perturbations
    )
    for leaf in jax.tree_util.tree_leaves(jac):
        assert jnp.all(jnp.isfinite(leaf))
