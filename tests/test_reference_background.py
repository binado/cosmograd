"""Backgrounds against CAMB over the shared list of cosmologies.

CAMB is run with ``mnu = 0`` (cosmograd folds massive neutrinos into
``omega_m``) and PPF dark energy for the w0wa models.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from .conftest import CosmologyCase, camb_results, cosmology_params

pytestmark = pytest.mark.slow

# Distances integrate over the whole history and agree to 4e-7. H, dV/dz and
# the inversion are sensitive to the radiation density at z ~ 1000, where
# cosmograd's rounded photon constant differs from CAMB's by 7e-5 in Omega_r
# and the agreement degrades to 1e-5.
RTOL_DISTANCE = 1e-6
RTOL_HIGH_Z = 5e-5
RTOL_RADIATION = 1e-4
Z = np.concatenate([np.linspace(0.01, 10.0, 60), [30.0, 100.0, 500.0, 1200.0]])


@pytest.mark.parametrize("case", cosmology_params())
def test_hubble_parameter(case: CosmologyCase) -> None:
    np.testing.assert_allclose(
        case.background.hubble_parameter(Z),
        camb_results(case).h_of_z(Z),
        rtol=RTOL_HIGH_Z,
    )


@pytest.mark.parametrize(
    ("method", "camb_method"),
    [
        ("comoving_distance", "comoving_radial_distance"),
        ("angular_diameter_distance", "angular_diameter_distance"),
        ("luminosity_distance", "luminosity_distance"),
    ],
)
@pytest.mark.parametrize("case", cosmology_params())
def test_distances(case: CosmologyCase, method: str, camb_method: str) -> None:
    expected = getattr(camb_results(case), camb_method)(Z)
    np.testing.assert_allclose(
        getattr(case.background, method)(Z), expected, rtol=RTOL_DISTANCE
    )


@pytest.mark.parametrize("case", cosmology_params())
def test_matter_density_parameter(case: CosmologyCase) -> None:
    bg = case.background
    expected = (
        bg.omega_m
        * (1 + Z) ** 3
        / (camb_results(case).h_of_z(Z) / bg.H0 * 299792.458) ** 2
    )
    np.testing.assert_allclose(
        bg.matter_density_parameter(Z), expected, rtol=RTOL_HIGH_Z
    )


@pytest.mark.parametrize("case", cosmology_params())
def test_differential_comoving_volume(case: CosmologyCase) -> None:
    # dV/dz/dΩ = D_M^2 / H with c = 1 and D_M the transverse comoving distance.
    results = camb_results(case)
    d_m = results.angular_diameter_distance(Z) * (1 + Z)
    expected = d_m**2 / results.h_of_z(Z)
    np.testing.assert_allclose(
        case.background.differential_comoving_volume(Z), expected, rtol=RTOL_HIGH_Z
    )


@pytest.mark.parametrize("case", cosmology_params())
def test_redshift_at_comoving_distance(case: CosmologyCase) -> None:
    chi = camb_results(case).comoving_radial_distance(Z)
    np.testing.assert_allclose(
        case.background.redshift_at_comoving_distance(jnp.asarray(chi)),
        Z,
        rtol=RTOL_HIGH_Z,
    )


@pytest.mark.parametrize("case", cosmology_params())
def test_radiation_density(case: CosmologyCase) -> None:
    results = camb_results(case)
    expected = results.get_Omega("photon") + results.get_Omega("neutrino")
    np.testing.assert_allclose(case.background.omega_r, expected, rtol=RTOL_RADIATION)
