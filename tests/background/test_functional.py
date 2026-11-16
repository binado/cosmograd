"""The pure functions on Hubble rates and parameters the package does not own."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from cosmograd import FlatLCDM
from cosmograd.background import functional
from cosmograd.constants import C_KM_S

from .reference import DeSitter, EinsteinDeSitter, Milne

Z = jnp.linspace(0.0, 5.0, 50)
H0 = 70.0


@pytest.fixture(
    params=[
        pytest.param((EinsteinDeSitter(H0=H0), 1.0, 0.0), id="einstein-de-sitter"),
        pytest.param((Milne(H0=H0), 0.0, 1.0), id="milne"),
        pytest.param((DeSitter(H0=H0), 0.0, 0.0), id="de-sitter"),
    ]
)
def universe(request: pytest.FixtureRequest) -> tuple[Any, float, float]:
    """A closed-form universe with its matter and curvature density parameters."""
    return request.param


def test_comoving_distance_matches_closed_form(
    universe: tuple[Any, float, float],
) -> None:
    exact, _, _ = universe
    np.testing.assert_allclose(
        functional.comoving_distance(type(exact).hubble_parameter, exact, Z),
        exact.comoving_distance(Z),
    )


def test_transverse_comoving_distance_matches_closed_form(
    universe: tuple[Any, float, float],
) -> None:
    exact, _, omega_k = universe
    np.testing.assert_allclose(
        functional.transverse_comoving_distance(
            type(exact).hubble_parameter, exact, Z, omega_k
        ),
        exact.transverse_comoving_distance(Z),
    )


def test_redshift_at_comoving_distance_matches_closed_form(
    universe: tuple[Any, float, float],
) -> None:
    exact, _, _ = universe
    chi = exact.comoving_distance(Z)
    np.testing.assert_allclose(
        functional.redshift_at_comoving_distance(
            type(exact).hubble_parameter, exact, chi
        ),
        Z,
        atol=1e-6,
    )


def test_matter_density_parameter_matches_closed_form(
    universe: tuple[Any, float, float],
) -> None:
    exact, omega_m, _ = universe
    np.testing.assert_allclose(
        functional.matter_density_parameter(
            type(exact).hubble_parameter, exact, Z, omega_m
        ),
        exact.matter_density_parameter(Z),
    )


def dict_hubble_parameter(params: dict[str, Array], z: Array) -> Array:
    """Flat, radiation-free ΛCDM read from a dict."""
    e2 = params["omega_m"] * (1 + z) ** 3 + 1 - params["omega_m"]
    return params["h"] * 100 / C_KM_S * jnp.sqrt(e2)


@pytest.fixture
def params() -> dict[str, Array]:
    return {"h": jnp.asarray(0.7), "omega_m": jnp.asarray(0.3)}


@pytest.fixture
def native(params: dict[str, Array]) -> FlatLCDM:
    """The native background of the same universe as ``params``."""
    return FlatLCDM(H0=100 * params["h"], omega_m=params["omega_m"], t_cmb=0.0)


def test_dict_parameters_match_native_background(
    params: dict[str, Array], native: FlatLCDM
) -> None:
    np.testing.assert_allclose(
        functional.comoving_distance(dict_hubble_parameter, params, Z),
        native.comoving_distance(Z),
    )
    np.testing.assert_allclose(
        functional.matter_density_parameter(
            dict_hubble_parameter, params, Z, params["omega_m"]
        ),
        native.matter_density_parameter(Z),
    )


def test_dict_parameter_gradients_match_native_background(
    params: dict[str, Array], native: FlatLCDM
) -> None:
    chi = native.comoving_distance(1.0)
    ours = jax.jit(
        jax.grad(
            lambda p: functional.redshift_at_comoving_distance(
                dict_hubble_parameter, p, chi
            )
        )
    )(params)
    expected = jax.grad(lambda c: c.redshift_at_comoving_distance(chi))(native)
    np.testing.assert_allclose(ours["h"], 100 * expected.H0)
    np.testing.assert_allclose(ours["omega_m"], expected.omega_m)


def test_dict_parameters_batch_under_vmap(params: dict[str, Array]) -> None:
    stacked = {"h": jnp.array([0.6, 0.7]), "omega_m": jnp.array([0.25, 0.3])}
    batched = jax.vmap(
        lambda p: functional.comoving_distance(dict_hubble_parameter, p, Z)
    )(stacked)
    np.testing.assert_allclose(
        batched[1], functional.comoving_distance(dict_hubble_parameter, params, Z)
    )
