"""``GenericBackground`` over a dict against the native background it describes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from cosmograd import LCDM, Background, GenericBackground
from cosmograd.constants import C_KM_S
from cosmograd.perturbations.growth import linear_growth

Z = jnp.linspace(0.0, 5.0, 25)
METHODS = [
    "hubble_parameter",
    "matter_density_parameter",
    "comoving_distance",
    "transverse_comoving_distance",
    "luminosity_distance",
    "angular_diameter_distance",
    "differential_comoving_volume",
]

Params = dict[str, Array]


def hubble(params: Params, z: Array) -> Array:
    """Radiation-free ΛCDM with curvature, read from a dict."""
    zp1 = 1 + z
    omega_m, omega_k = params["omega_m"], params["omega_k"]
    e2 = omega_m * zp1**3 + omega_k * zp1**2 + 1 - omega_m - omega_k
    return params["h"] * 100 / C_KM_S * jnp.sqrt(e2)


def omega_m(params: Params) -> Array:
    return params["omega_m"]


def omega_k(params: Params) -> Array:
    return params["omega_k"]


@pytest.fixture(params=[0.0, 0.05, -0.05], ids=["flat", "open", "closed"])
def params(request: pytest.FixtureRequest) -> Params:
    return {
        "h": jnp.asarray(0.7),
        "omega_m": jnp.asarray(0.3),
        "omega_k": jnp.asarray(request.param),
    }


@pytest.fixture
def generic(params: Params) -> GenericBackground[Params]:
    return GenericBackground(params, hubble, omega_m, omega_k)


@pytest.fixture
def native(params: Params) -> LCDM:
    return LCDM(
        H0=100 * params["h"],
        omega_m=params["omega_m"],
        omega_k=params["omega_k"],
        t_cmb=0.0,
    )


def test_is_a_background(generic: GenericBackground[Params]) -> None:
    assert isinstance(generic, Background)


@pytest.mark.parametrize("method", METHODS)
def test_method_matches_native_background(
    generic: GenericBackground[Params], native: LCDM, method: str
) -> None:
    np.testing.assert_allclose(
        jax.jit(lambda b: getattr(b, method)(Z))(generic), getattr(native, method)(Z)
    )


def test_hubble_constant_matches_native_background(
    generic: GenericBackground[Params], native: LCDM
) -> None:
    np.testing.assert_allclose(generic.hubble_constant, native.hubble_constant)


def test_redshift_at_comoving_distance_matches_native_background(
    generic: GenericBackground[Params], native: LCDM
) -> None:
    chi = native.comoving_distance(Z)
    np.testing.assert_allclose(
        generic.redshift_at_comoving_distance(chi),
        native.redshift_at_comoving_distance(chi),
        atol=1e-6,
    )


def test_gradient_comes_back_in_the_parameter_layout(
    generic: GenericBackground[Params], native: LCDM
) -> None:
    ours = jax.grad(lambda b: b.luminosity_distance(1.0))(generic).params
    expected = jax.grad(lambda c: c.luminosity_distance(1.0))(native)
    np.testing.assert_allclose(ours["h"], 100 * expected.H0)
    np.testing.assert_allclose(ours["omega_m"], expected.omega_m)
    np.testing.assert_allclose(ours["omega_k"], expected.omega_k)


def test_omega_k_defaults_to_flat(params: Params) -> None:
    flat = GenericBackground(params, hubble, omega_m)
    np.testing.assert_allclose(
        flat.transverse_comoving_distance(Z), flat.comoving_distance(Z)
    )


def test_linear_growth_matches_native_background(
    generic: GenericBackground[Params], native: LCDM
) -> None:
    np.testing.assert_allclose(
        linear_growth(generic, Z), linear_growth(native, Z), rtol=1e-6
    )
