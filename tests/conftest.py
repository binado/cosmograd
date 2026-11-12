"""Test configuration. The package itself never changes JAX precision."""

import importlib.util

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from cosmograd import FlatLCDM, Perturbations
from cosmograd.background import Background
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

jax.config.update("jax_enable_x64", True)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # the parent package of a dotted name is missing
        return False


def _requires(module: str) -> pytest.MarkDecorator:
    """Skip a backend whose optional dependency is not installed."""
    return pytest.mark.skipif(
        not _installed(module),
        reason=f"optional dependency {module!r} is not installed",
    )


@pytest.fixture(params=[FlatLCDM(h=0.67, omega_m=0.31)], ids=lambda c: type(c).__name__)
def background(request: pytest.FixtureRequest) -> Background:
    """Every in-tree background implementation; add new backends to ``params``."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            CosmoPower(h=0.67, omega_m=0.31, omega_b=0.049, n_s=0.965, ln10as=3.044),
            id="CosmoPower",
            marks=_requires("cosmopower_jax"),
        ),
        pytest.param(
            BaccoEmu(
                h=0.67, omega_cold=0.31, omega_baryon=0.049, ns=0.965, sigma8_cold=0.81
            ),
            id="BaccoEmu",
            marks=_requires("baccoemu.matter_powerspectrum"),
        ),
    ]
)
def perturbations(request: pytest.FixtureRequest) -> Perturbations:
    """Every in-tree perturbations implementation; add new backends to ``params``.

    Emulator backends are skipped when their optional extra is missing.
    """
    return request.param


# Wavenumbers (1/Mpc) and redshifts inside each backend's documented range.
_DEFAULT_GRID = (jnp.geomspace(1e-3, 1.0, 7), jnp.linspace(0.0, 3.0, 4))
_GRIDS: dict[type, tuple[Array, Array]] = {
    BaccoEmu: (jnp.geomspace(1e-3, 1.0, 7), jnp.linspace(0.0, 1.5, 4)),
}


@pytest.fixture
def perturbations_grid(perturbations: Perturbations) -> tuple[Array, Array]:
    """1-D ``(k, z)`` inside the range the ``perturbations`` backend documents."""
    return _GRIDS.get(type(perturbations), _DEFAULT_GRID)
