"""Physical expectations every perturbations backend should meet.

The protocol contract itself is checked by
:func:`cosmograd.testing.check_perturbations` in ``test_conformance.py``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .conftest import PerturbationsCase


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


def test_nonlinear_mode_is_static(perturbations_case: PerturbationsCase) -> None:
    if not perturbations_case.supports_nonlinear:
        pytest.skip("backend supports linear power only")
    linear = replace(perturbations_case.cosmo, nonlinear=False)
    nonlinear = replace(perturbations_case.cosmo, nonlinear=True)
    assert jax.tree.structure(linear) != jax.tree.structure(nonlinear)
    np.testing.assert_array_equal(jax.tree.leaves(linear), jax.tree.leaves(nonlinear))
    k, z = 1.0, 0.0
    assert nonlinear.matter_power_spectrum(k, z) > linear.matter_power_spectrum(k, z)
