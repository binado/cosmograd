import jax.numpy as jnp
import pytest

from cosmograd.background import Background
from cosmograd.testing import check_background


@pytest.mark.parametrize(
    "z", [jnp.linspace(0.0, 5.0, 50), jnp.array(1.0)], ids=["vector", "scalar"]
)
def test_background_conformance(background: Background, z: jnp.ndarray) -> None:
    check_background(background, z)
