"""Test configuration. The package itself never changes JAX precision."""

import importlib.util

import jax
import pytest

from cosmograd import FlatLCDM, Perturbations
from cosmograd.background import Background
from cosmograd.perturbations.cosmopower import CosmoPower

jax.config.update("jax_enable_x64", True)


def _requires(module: str) -> pytest.MarkDecorator:
    """Skip a backend whose optional dependency is not installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec(module) is None,
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
    ]
)
def perturbations(request: pytest.FixtureRequest) -> Perturbations:
    """Every in-tree perturbations implementation; add new backends to ``params``.

    Emulator backends are skipped when their optional extra is missing.
    """
    return request.param
