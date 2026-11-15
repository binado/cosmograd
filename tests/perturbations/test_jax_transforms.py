"""Perturbation methods behave under ``jit``, ``vmap`` and ``grad`` as expected."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .conftest import PerturbationsCase


def test_leaves_are_floating_point(perturbations_case: PerturbationsCase) -> None:
    leaves = jax.tree.leaves(perturbations_case.cosmo)
    assert leaves
    for leaf in leaves:
        assert jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)


@pytest.mark.parametrize(
    "method", ["matter_power_spectrum", "linear_matter_power_spectrum"]
)
def test_power_spectrum_under_jit_matches_eager(
    perturbations_case: PerturbationsCase, method: str
) -> None:
    cosmo, k, z = perturbations_case.cosmo, perturbations_case.k, perturbations_case.z
    np.testing.assert_allclose(
        jax.jit(lambda c: getattr(c, method)(k[:, None], z))(cosmo),
        getattr(cosmo, method)(k[:, None], z),
    )


@pytest.mark.parametrize("method", ["growth_function", "growth_rate"])
@pytest.mark.parametrize(
    "with_k", [False, True], ids=["large-scale", "scale-dependent"]
)
def test_growth_under_jit_matches_eager(
    perturbations_case: PerturbationsCase, method: str, with_k: bool
) -> None:
    cosmo, z = perturbations_case.cosmo, perturbations_case.z
    k = perturbations_case.k[:, None] if with_k else None
    np.testing.assert_allclose(
        jax.jit(lambda c: getattr(c, method)(z, k=k))(cosmo),
        getattr(cosmo, method)(z, k=k),
    )


def test_power_spectrum_under_vmap_over_cosmologies_matches_loop(
    perturbations_case: PerturbationsCase,
) -> None:
    first, k, z = perturbations_case.cosmo, perturbations_case.k, perturbations_case.z
    background = first.background
    second = replace(
        first, background=replace(background, omega_m=background.omega_m + 0.01)
    )
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), first, second)
    actual = jax.jit(jax.vmap(lambda c: c.matter_power_spectrum(k[:, None], z)))(batch)
    expected = [c.matter_power_spectrum(k[:, None], z) for c in (first, second)]
    # Float32 emulator networks can round differently under batched evaluation.
    np.testing.assert_allclose(actual, np.array(expected), rtol=1e-5)


def test_log_power_spectrum_has_finite_nonzero_gradient_in_every_parameter(
    perturbations_case: PerturbationsCase,
) -> None:
    cosmo = perturbations_case.cosmo
    gradient = jax.jit(jax.grad(lambda c: jnp.log(c.matter_power_spectrum(0.1, 0.5))))(
        cosmo
    )
    leaves = jnp.array(jax.tree.leaves(gradient.background))
    assert jnp.all(jnp.isfinite(leaves) & (leaves != 0))
