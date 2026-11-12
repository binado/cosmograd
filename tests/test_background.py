from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import cosmology as astropy_cosmology
from astropy import units as u

from cosmograd import FlatLCDM

H, OMEGA_M, T_CMB, N_EFF = 0.6736, 0.3153, 2.7255, 3.044
Z = np.linspace(0.0, 10.0, 101)
RTOL = 1e-5

# astropy creates its cosmology classes dynamically, so static checkers cannot
# see their constructor signatures or use them as annotations.
FlatLambdaCDM: Any = astropy_cosmology.FlatLambdaCDM


@pytest.fixture(scope="module")
def cosmo() -> FlatLCDM:
    return FlatLCDM(h=H, omega_m=OMEGA_M, t_cmb=T_CMB, n_eff=N_EFF)


@pytest.fixture(scope="module")
def reference() -> Any:
    return FlatLambdaCDM(
        H0=100 * H, Om0=OMEGA_M, Tcmb0=T_CMB, Neff=N_EFF, m_nu=0.0 * u.eV
    )


def test_hubble_constant(cosmo: FlatLCDM, reference: Any) -> None:
    expected = (reference.H0 / reference.H0.unit).value / 299792.458
    np.testing.assert_allclose(cosmo.hubble_constant, expected, rtol=1e-12)


def test_radiation(cosmo: FlatLCDM, reference: Any) -> None:
    expected = reference.Ogamma0 + reference.Onu0
    np.testing.assert_allclose(cosmo.omega_r, expected, rtol=1e-4)


@pytest.mark.parametrize(
    ("method", "reference_method", "unit"),
    [
        ("hubble_parameter", "H", u.km / u.s / u.Mpc),
        ("comoving_distance", "comoving_distance", u.Mpc),
        ("luminosity_distance", "luminosity_distance", u.Mpc),
        ("angular_diameter_distance", "angular_diameter_distance", u.Mpc),
        (
            "differential_comoving_volume",
            "differential_comoving_volume",
            u.Mpc**3 / u.sr,
        ),
    ],
)
def test_against_astropy(
    cosmo: FlatLCDM,
    reference: Any,
    method: str,
    reference_method: str,
    unit: u.UnitBase,
) -> None:
    expected = getattr(reference, reference_method)(Z).to_value(unit)
    if method == "hubble_parameter":
        expected = expected / 299792.458
    np.testing.assert_allclose(getattr(cosmo, method)(Z), expected, rtol=RTOL)


def test_matter_density_parameter(cosmo: FlatLCDM, reference: Any) -> None:
    np.testing.assert_allclose(
        cosmo.matter_density_parameter(Z), reference.Om(Z), rtol=RTOL
    )


def test_inverse(cosmo: FlatLCDM) -> None:
    z = jnp.linspace(0.01, 1000.0, 200)
    np.testing.assert_allclose(
        cosmo.redshift_at_comoving_distance(cosmo.comoving_distance(z)), z, rtol=RTOL
    )


def test_leaves_are_parameters(cosmo: FlatLCDM) -> None:
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves == [H, OMEGA_M]


def test_gradient_matches_finite_difference(cosmo: FlatLCDM) -> None:
    z = 1.0
    grad = jax.grad(lambda c: c.luminosity_distance(z))(cosmo)
    eps = 1e-6
    for name in ("h", "omega_m"):
        up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
        down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
        fd = (up.luminosity_distance(z) - down.luminosity_distance(z)) / (2 * eps)
        np.testing.assert_allclose(getattr(grad, name), fd, rtol=1e-5)


def test_vmap_over_cosmologies() -> None:
    batch = FlatLCDM(h=jnp.array([0.6, 0.7]), omega_m=jnp.array([0.3, 0.3]))
    chi = jax.vmap(lambda c: c.comoving_distance(1.0))(batch)
    assert chi.shape == (2,)
    # Larger h means a shorter distance in Mpc.
    assert chi[0] > chi[1]
