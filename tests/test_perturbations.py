"""Physical expectations every perturbations backend should meet.

The protocol contract itself is checked by
:func:`cosmograd.testing.check_perturbations` in ``test_conformance.py``.
"""

import jax.numpy as jnp
import numpy as np

from cosmograd import Perturbations

from .conftest import PerturbationsCase


def test_growth_decreases_with_redshift(perturbations_case: PerturbationsCase) -> None:
    growth = perturbations_case.cosmo.growth_function(perturbations_case.z)
    assert jnp.all(jnp.diff(growth) < 0)


def test_nonlinear_matches_linear_on_large_scales(
    perturbations: Perturbations,
) -> None:
    k, z = jnp.geomspace(1e-4, 1e-3, 5), 0.5
    np.testing.assert_allclose(
        perturbations.matter_power_spectrum(k, z, nonlinear=True),
        perturbations.matter_power_spectrum(k, z),
        rtol=1e-2,
    )
