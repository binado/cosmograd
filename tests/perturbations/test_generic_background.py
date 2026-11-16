"""Every backend on a dict behind ``GenericBackground`` against the native cosmology."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from cosmograd import EisensteinHu, FlatLCDM, GenericBackground, Perturbations
from cosmograd.perturbations import EisensteinHuParameters
from cosmograd.perturbations.baccoemu import BaccoEmu, BaccoEmuParameters
from cosmograd.perturbations.cosmopower import CosmoPower, CosmoPowerParameters

from ..cosmologies import EMULATOR_COSMOLOGY
from ..optional import requires

Params = dict[str, Array]
Generic = GenericBackground[Params]

K = jnp.geomspace(1e-3, 1.0, 5)
Z = jnp.array([0.0, 0.5, 1.5])
# The bacco networks evaluate in float32.
RTOL = 1e-5


def hubble(params: Params, z: Array) -> Array:
    return FlatLCDM(H0=params["H0"], omega_m=params["omega_m"]).hubble_parameter(z)


def omega_m(params: Params) -> Array:
    return params["omega_m"]


def eisenstein_hu_parameters(background: Generic) -> EisensteinHuParameters:
    p = background.params
    return EisensteinHuParameters(
        p["omega_b"], p["n_s"], p["ln10as"], EMULATOR_COSMOLOGY.t_cmb
    )


def cosmopower_parameters(background: Generic) -> CosmoPowerParameters:
    p = background.params
    return CosmoPowerParameters(p["omega_b"], p["n_s"], p["ln10as"])


def baccoemu_parameters(background: Generic) -> BaccoEmuParameters:
    p = background.params
    return BaccoEmuParameters(p["omega_b"], p["n_s"], p["ln10as"])


class Pair(NamedTuple):
    """One backend on the dict-backed background and on the native cosmology."""

    generic: Any
    native: Any


@pytest.fixture
def background() -> Generic:
    """``EMULATOR_COSMOLOGY`` held as a dict."""
    params = {
        name: jnp.asarray(getattr(EMULATOR_COSMOLOGY, name))
        for name in ("H0", "omega_m", "omega_b", "n_s", "ln10as")
    }
    return GenericBackground(params, hubble, omega_m)


@pytest.fixture(
    params=[
        pytest.param((EisensteinHu, eisenstein_hu_parameters), id="eisenstein-hu"),
        pytest.param(
            (CosmoPower, cosmopower_parameters),
            id="cosmopower",
            marks=requires("cosmopower_jax"),
        ),
        pytest.param(
            (BaccoEmu, baccoemu_parameters),
            id="baccoemu",
            marks=requires("baccoemu.matter_powerspectrum"),
        ),
    ]
)
def pair(request: pytest.FixtureRequest, background: Generic) -> Pair:
    backend, parameters = request.param
    return Pair(backend(background, parameters=parameters), backend(EMULATOR_COSMOLOGY))


def test_is_perturbations(pair: Pair) -> None:
    assert isinstance(pair.generic, Perturbations)


@pytest.mark.parametrize(
    "quantity",
    [
        lambda c: c.linear_matter_power_spectrum(K, Z[:, None]),
        lambda c: c.growth_function(Z),
        lambda c: c.growth_rate(Z),
        lambda c: c.primordial_power_spectrum(K),
    ],
    ids=[
        "linear_matter_power_spectrum",
        "growth_function",
        "growth_rate",
        "primordial",
    ],
)
def test_quantity_matches_native_cosmology(
    pair: Pair, quantity: Callable[[Any], Array]
) -> None:
    np.testing.assert_allclose(
        jax.jit(quantity)(pair.generic), quantity(pair.native), rtol=RTOL
    )


def test_gradient_comes_back_in_the_parameter_layout(pair: Pair) -> None:
    def log_power(cosmo: Any) -> Array:
        return jnp.log(cosmo.linear_matter_power_spectrum(0.1, 0.5))

    ours = jax.grad(log_power)(pair.generic).background.params
    expected = jax.grad(log_power)(pair.native).background
    for name, value in ours.items():
        np.testing.assert_allclose(value, getattr(expected, name), rtol=1e-4)
