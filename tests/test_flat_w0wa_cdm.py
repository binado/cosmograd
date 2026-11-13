"""CPL background against astropy and its ΛCDM limit."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import cosmology as astropy_cosmology
from astropy import units as u

from cosmograd import FlatLCDM, FlatW0WaCDM

H0, OMEGA_M = 67.36, 0.3153
Z = np.linspace(0.0, 10.0, 101)
RTOL = 1e-5

# astropy creates its cosmology classes dynamically; see test_background.py.
AstropyFlatw0waCDM: Any = astropy_cosmology.Flatw0waCDM


@pytest.fixture(
    params=[(-1.0, 0.0), (-0.8, 0.0), (-0.9, 0.3), (-1.2, -0.4)],
    ids=["lambda", "constant-w", "thawing", "phantom"],
    scope="module",
)
def cosmo(request: pytest.FixtureRequest) -> FlatW0WaCDM:
    w0, wa = request.param
    return FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=w0, wa=wa)


@pytest.fixture(scope="module")
def reference(cosmo: FlatW0WaCDM) -> Any:
    return AstropyFlatw0waCDM(
        H0=H0,
        Om0=OMEGA_M,
        w0=cosmo.w0,
        wa=cosmo.wa,
        Tcmb0=cosmo.t_cmb,
        Neff=cosmo.n_eff,
        m_nu=0.0 * u.eV,
    )


@pytest.mark.parametrize(
    ("method", "reference_method", "unit"),
    [
        ("efunc", "efunc", None),
        ("matter_density_parameter", "Om", None),
        ("comoving_distance", "comoving_distance", u.Mpc),
        ("luminosity_distance", "luminosity_distance", u.Mpc),
    ],
)
def test_against_astropy(
    cosmo: FlatW0WaCDM,
    reference: Any,
    method: str,
    reference_method: str,
    unit: u.UnitBase | None,
) -> None:
    expected = getattr(reference, reference_method)(Z)
    if unit is not None:
        expected = expected.to_value(unit)
    np.testing.assert_allclose(getattr(cosmo, method)(Z), expected, rtol=RTOL)


def test_inverse_round_trip(cosmo: FlatW0WaCDM) -> None:
    z = jnp.array([[0.0, 1e-8, 0.1], [1.0, 100.0, 1200.0]])
    inverse = jax.jit(cosmo.redshift_at_comoving_distance)
    np.testing.assert_allclose(inverse(cosmo.comoving_distance(z)), z)


@pytest.mark.parametrize(
    "method", ["hubble_parameter", "comoving_distance", "matter_density_parameter"]
)
def test_defaults_match_flat_lcdm(method: str) -> None:
    lcdm = FlatLCDM(H0=H0, omega_m=OMEGA_M)
    w0wa = FlatW0WaCDM(H0=H0, omega_m=OMEGA_M)
    np.testing.assert_allclose(getattr(w0wa, method)(Z), getattr(lcdm, method)(Z))


def test_leaves_are_parameters(cosmo: FlatW0WaCDM) -> None:
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves == [H0, OMEGA_M, cosmo.w0, cosmo.wa]


@pytest.mark.parametrize("name", ["H0", "omega_m", "w0", "wa"])
def test_gradient_matches_finite_difference(cosmo: FlatW0WaCDM, name: str) -> None:
    z = 1.0
    grad = jax.grad(lambda c: c.luminosity_distance(z))(cosmo)
    eps = 1e-6
    up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
    down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
    fd = (up.luminosity_distance(z) - down.luminosity_distance(z)) / (2 * eps)
    np.testing.assert_allclose(getattr(grad, name), fd, rtol=RTOL)
