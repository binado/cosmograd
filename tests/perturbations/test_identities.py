"""Relations every perturbations backend satisfies, linear or nonlinear."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import Background, Perturbations

from .conftest import PerturbationsCase


def test_backend_provides_the_perturbations_and_background_protocols(
    perturbations: Perturbations,
) -> None:
    assert isinstance(perturbations, Perturbations)
    assert isinstance(perturbations.background, Background)


def test_power_spectra_are_finite_and_positive(
    perturbations_case: PerturbationsCase,
) -> None:
    cosmo, k, z = perturbations_case.cosmo, perturbations_case.k, perturbations_case.z
    for pk in (
        cosmo.matter_power_spectrum(k[:, None], z),
        cosmo.linear_matter_power_spectrum(k[:, None], z),
        cosmo.primordial_power_spectrum(k),
    ):
        assert jnp.all(jnp.isfinite(pk) & (pk > 0))


def test_power_spectrum_on_outer_grid_matches_elementwise_calls(
    perturbations_case: PerturbationsCase,
) -> None:
    cosmo, k, z = perturbations_case.cosmo, perturbations_case.k, perturbations_case.z
    outer = cosmo.matter_power_spectrum(k[:, None], z[None, :])
    k_b, z_b = jnp.broadcast_arrays(k[:, None], z[None, :])
    elementwise = cosmo.matter_power_spectrum(k_b.ravel(), z_b.ravel())
    np.testing.assert_allclose(outer.ravel(), elementwise)


def test_growth_function_is_one_today(perturbations_case: PerturbationsCase) -> None:
    cosmo, k = perturbations_case.cosmo, perturbations_case.k
    np.testing.assert_allclose(cosmo.growth_function(0.0, k=k), jnp.ones_like(k))
    np.testing.assert_allclose(cosmo.growth_function(0.0), 1.0)


def test_growth_function_squared_is_the_ratio_of_linear_power_spectra(
    perturbations_case: PerturbationsCase,
) -> None:
    cosmo, k, z = perturbations_case.cosmo, perturbations_case.k, perturbations_case.z
    k = k[:, None]
    np.testing.assert_allclose(
        cosmo.growth_function(z, k=k) ** 2,
        cosmo.linear_matter_power_spectrum(k, z)
        / cosmo.linear_matter_power_spectrum(k, 0.0),
    )


def test_growth_rate_is_logarithmic_derivative_of_growth_function(
    perturbations_case: PerturbationsCase,
) -> None:
    cosmo, z = perturbations_case.cosmo, perturbations_case.z
    derivative = jax.vmap(jax.grad(lambda x: jnp.log(cosmo.growth_function(x))))(z)
    np.testing.assert_allclose(
        cosmo.growth_rate(z),
        -(1 + z) * derivative,
        rtol=perturbations_case.growth_rate_rtol,
    )


def test_linear_power_spectrum_is_proportional_to_primordial_amplitude(
    perturbations_case: PerturbationsCase,
) -> None:
    """:math:`P \\propto A_s`, so :math:`\\mathrm{d} \\ln P / \\mathrm{d} \\ln(10^{10} A_s) = 1`."""
    cosmo = perturbations_case.cosmo
    grad = jax.grad(lambda c: jnp.log(c.linear_matter_power_spectrum(0.1, 0.5)))(cosmo)
    np.testing.assert_allclose(grad.background.ln10as, 1.0, rtol=1e-2)


def test_growth_is_independent_of_nonlinear_mode(
    perturbations_case: PerturbationsCase,
) -> None:
    linear = replace(perturbations_case.cosmo, nonlinear=False)
    k, z = perturbations_case.k[:, None], perturbations_case.z
    np.testing.assert_allclose(
        perturbations_case.cosmo.growth_function(z, k=k),
        linear.growth_function(z, k=k),
    )
    np.testing.assert_allclose(
        perturbations_case.cosmo.growth_rate(z, k=k),
        linear.growth_rate(z, k=k),
    )


def test_linear_power_is_independent_of_nonlinear_mode(
    perturbations_case: PerturbationsCase,
) -> None:
    linear = replace(perturbations_case.cosmo, nonlinear=False)
    k, z = perturbations_case.k[:, None], perturbations_case.z
    np.testing.assert_allclose(
        perturbations_case.cosmo.linear_matter_power_spectrum(k, z),
        linear.matter_power_spectrum(k, z),
    )


def test_growth_decreases_with_redshift(perturbations_case: PerturbationsCase) -> None:
    growth = perturbations_case.cosmo.growth_function(perturbations_case.z)
    assert jnp.all(jnp.diff(growth) < 0)


def test_nonlinear_matches_linear_on_large_scales(
    perturbations_case: PerturbationsCase,
) -> None:
    if not perturbations_case.supports_nonlinear:
        pytest.skip("backend supports linear power only")
    perturbations = perturbations_case.cosmo
    k, z = jnp.geomspace(1e-4, 1e-3, 5), 0.5
    np.testing.assert_allclose(
        replace(perturbations, nonlinear=True).matter_power_spectrum(k, z),
        replace(perturbations, nonlinear=False).matter_power_spectrum(k, z),
        rtol=1e-2,
    )


def test_nonlinear_mode_is_static_and_adds_power_on_small_scales(
    perturbations_case: PerturbationsCase,
) -> None:
    if not perturbations_case.supports_nonlinear:
        pytest.skip("backend supports linear power only")
    linear = replace(perturbations_case.cosmo, nonlinear=False)
    nonlinear = replace(perturbations_case.cosmo, nonlinear=True)
    assert jax.tree.structure(linear) != jax.tree.structure(nonlinear)
    np.testing.assert_array_equal(jax.tree.leaves(linear), jax.tree.leaves(nonlinear))
    k, z = 1.0, 0.0
    assert nonlinear.matter_power_spectrum(k, z) > linear.matter_power_spectrum(k, z)
