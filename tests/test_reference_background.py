"""Backgrounds against CAMB over the shared list of cosmologies.

CAMB is run with ``mnu = 0`` (cosmograd folds massive neutrinos into
``omega_m``) and PPF dark energy for the w0wa models.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from .conftest import camb_results
from .cosmologies import Cosmology, cosmology_params

pytestmark = pytest.mark.slow

# Direct evaluations agree to ~1e-9. The inversion is a root solve (3e-8) and
# omega_r is limited by the digits kept in the photon constant (1e-8).
RTOL = 1e-8
RTOL_LOOSE = 1e-7
Z = np.concatenate([np.linspace(0.01, 10.0, 60), [30.0, 100.0, 500.0, 1200.0]])


@pytest.mark.parametrize("case", cosmology_params())
def test_hubble_parameter(case: Cosmology) -> None:
    np.testing.assert_allclose(
        case.hubble_parameter(Z),
        camb_results(case).h_of_z(Z),
        rtol=RTOL,
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
def test_distances(case: Cosmology, method: str, camb_method: str) -> None:
    expected = getattr(camb_results(case), camb_method)(Z)
    np.testing.assert_allclose(getattr(case, method)(Z), expected, rtol=RTOL)


@pytest.mark.parametrize("case", cosmology_params())
def test_matter_density_parameter(case: Cosmology) -> None:
    bg = case
    expected = (
        bg.omega_m
        * (1 + Z) ** 3
        / (camb_results(case).h_of_z(Z) / bg.H0 * 299792.458) ** 2
    )
    np.testing.assert_allclose(bg.matter_density_parameter(Z), expected, rtol=RTOL)


@pytest.mark.parametrize("case", cosmology_params())
def test_differential_comoving_volume(case: Cosmology) -> None:
    # dV/dz/dΩ = D_M^2 / H with c = 1 and D_M the transverse comoving distance.
    results = camb_results(case)
    d_m = results.angular_diameter_distance(Z) * (1 + Z)
    expected = d_m**2 / results.h_of_z(Z)
    np.testing.assert_allclose(
        case.differential_comoving_volume(Z), expected, rtol=RTOL
    )


@pytest.mark.parametrize("case", cosmology_params())
def test_redshift_at_comoving_distance(case: Cosmology) -> None:
    chi = camb_results(case).comoving_radial_distance(Z)
    np.testing.assert_allclose(
        case.redshift_at_comoving_distance(jnp.asarray(chi)),
        Z,
        rtol=RTOL_LOOSE,
    )


@pytest.mark.parametrize("case", cosmology_params())
def test_radiation_density(case: Cosmology) -> None:
    results = camb_results(case)
    expected = results.get_Omega("photon") + results.get_Omega("neutrino")
    np.testing.assert_allclose(case.omega_r, expected, rtol=RTOL_LOOSE)
