"""Test configuration. The package itself never changes JAX precision."""

import jax
import pytest

from cosmograd import FlatLCDM
from cosmograd.background import Background

jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[FlatLCDM(h=0.67, omega_m=0.31)], ids=lambda c: type(c).__name__)
def background(request: pytest.FixtureRequest) -> Background:
    """Every in-tree background implementation; add new backends to ``params``."""
    return request.param
