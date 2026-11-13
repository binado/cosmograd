"""Curved and general backgrounds against astropy and their constrained limits."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import cosmology as astropy_cosmology
from astropy import units as u

from cosmograd import LCDM, FlatLCDM, FlatW0WaCDM, W0WaCDM

H0, OMEGA_M = 67.36, 0.3153
Z = np.linspace(0.0, 10.0, 101)
RTOL = 1e-5

# astropy creates its cosmology classes dynamically; see test_background.py.
AstropyW0waCDM: Any = astropy_cosmology.w0waCDM


@pytest.fixture(
    params=[
        (-0.1, -1.0, 0.0),
        (-0.02, -1.0, 0.0),
        (0.05, -1.0, 0.0),
        (0.1, -0.9, 0.3),
        (-0.05, -1.2, -0.4),
    ],
    ids=["closed", "nearly-flat-closed", "open", "open-thawing", "closed-phantom"],
)
def cosmo(request: pytest.FixtureRequest) -> W0WaCDM:
    omega_k, w0, wa = request.param
    return W0WaCDM(H0=H0, omega_m=OMEGA_M, omega_k=omega_k, w0=w0, wa=wa)


@pytest.fixture
def reference(cosmo: W0WaCDM) -> Any:
    # astropy fixes Omega_k = 1 - Omega_m - Omega_de - Omega_r.
    return AstropyW0waCDM(
        H0=H0,
        Om0=OMEGA_M,
        Ode0=float(cosmo.omega_de),
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
        ("comoving_distance", "comoving_distance", u.Mpc),
        ("transverse_comoving_distance", "comoving_transverse_distance", u.Mpc),
        ("luminosity_distance", "luminosity_distance", u.Mpc),
        ("angular_diameter_distance", "angular_diameter_distance", u.Mpc),
    ],
)
def test_w0wacdm_curved_matches_astropy(
    cosmo: W0WaCDM,
    reference: Any,
    method: str,
    reference_method: str,
    unit: u.UnitBase | None,
) -> None:
    expected = getattr(reference, reference_method)(Z)
    if unit is not None:
        expected = expected.to_value(unit)
    np.testing.assert_allclose(getattr(cosmo, method)(Z), expected, rtol=RTOL)


def test_w0wacdm_curved_inverse_recovers_redshift(cosmo: W0WaCDM) -> None:
    z = jnp.array([[0.0, 1e-8, 0.1], [1.0, 100.0, 1200.0]])
    inverse = jax.jit(cosmo.redshift_at_comoving_distance)
    np.testing.assert_allclose(inverse(cosmo.comoving_distance(z)), z)


@pytest.mark.parametrize(
    ("omega_k", "expected_sign"), [(-0.1, -1), (0.1, 1)], ids=["closed", "open"]
)
def test_w0wacdm_curvature_bends_distance_from_comoving(
    omega_k: float, expected_sign: int
) -> None:
    cosmo = W0WaCDM(H0=H0, omega_m=OMEGA_M, omega_k=omega_k)
    gap = cosmo.transverse_comoving_distance(2.0) - cosmo.comoving_distance(2.0)
    assert np.sign(gap) == expected_sign


@pytest.mark.parametrize("omega_k", [0.0, 1e-8, -1e-8, 1e-3])
def test_w0wacdm_curvature_gradient_is_finite_near_flat(omega_k: float) -> None:
    cosmo = W0WaCDM(H0=H0, omega_m=OMEGA_M, omega_k=omega_k)
    grad = jax.grad(lambda c: c.luminosity_distance(1.0))(cosmo)
    assert np.isfinite(grad.omega_k)


@pytest.mark.parametrize("name", ["H0", "omega_m", "omega_k", "w0", "wa"])
def test_w0wacdm_gradient_matches_finite_difference(cosmo: W0WaCDM, name: str) -> None:
    z = 1.0
    grad = jax.grad(lambda c: c.luminosity_distance(z))(cosmo)
    eps = 1e-6
    up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
    down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
    fd = (up.luminosity_distance(z) - down.luminosity_distance(z)) / (2 * eps)
    np.testing.assert_allclose(getattr(grad, name), fd, rtol=RTOL)


def test_w0wacdm_leaves_are_parameters(cosmo: W0WaCDM) -> None:
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves == [H0, OMEGA_M, cosmo.omega_k, cosmo.w0, cosmo.wa]


@pytest.mark.parametrize(
    ("constrained", "general"),
    [
        (
            FlatLCDM(H0=H0, omega_m=OMEGA_M),
            W0WaCDM(H0=H0, omega_m=OMEGA_M),
        ),
        (
            FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=-0.9, wa=0.3),
            W0WaCDM(H0=H0, omega_m=OMEGA_M, w0=-0.9, wa=0.3),
        ),
        (
            LCDM(H0=H0, omega_m=OMEGA_M, omega_k=0.05),
            W0WaCDM(H0=H0, omega_m=OMEGA_M, omega_k=0.05),
        ),
    ],
    ids=["flat-lcdm", "flat-w0wacdm", "lcdm"],
)
@pytest.mark.parametrize(
    "method",
    [
        "efunc",
        "matter_density_parameter",
        "comoving_distance",
        "transverse_comoving_distance",
        "luminosity_distance",
    ],
)
def test_constrained_class_matches_general_at_fixed_parameters(
    constrained: Any, general: W0WaCDM, method: str
) -> None:
    np.testing.assert_allclose(
        getattr(constrained, method)(Z), getattr(general, method)(Z), rtol=RTOL
    )


@pytest.mark.parametrize(
    ("constrained", "leaves"),
    [
        (FlatLCDM(H0=H0, omega_m=OMEGA_M), [H0, OMEGA_M]),
        (LCDM(H0=H0, omega_m=OMEGA_M, omega_k=0.05), [H0, OMEGA_M, 0.05]),
        (
            FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=-0.9, wa=0.3),
            [H0, OMEGA_M, -0.9, 0.3],
        ),
    ],
    ids=["flat-lcdm", "lcdm", "flat-w0wacdm"],
)
def test_constrained_class_leaves_exclude_fixed_parameters(
    constrained: Any, leaves: list[float]
) -> None:
    assert jax.tree_util.tree_leaves(constrained) == leaves


@pytest.mark.parametrize(
    ("constrained", "fixed"),
    [
        (FlatLCDM(H0=H0, omega_m=OMEGA_M), {"omega_k": 0.0, "w0": -1.0, "wa": 0.0}),
        (FlatW0WaCDM(H0=H0, omega_m=OMEGA_M), {"omega_k": 0.0}),
        (LCDM(H0=H0, omega_m=OMEGA_M), {"w0": -1.0, "wa": 0.0}),
    ],
    ids=["flat-lcdm", "flat-w0wacdm", "lcdm"],
)
def test_constrained_class_fixed_parameters_have_expected_values(
    constrained: Any, fixed: dict[str, float]
) -> None:
    assert {name: getattr(constrained, name) for name in fixed} == fixed
