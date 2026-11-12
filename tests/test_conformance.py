import jax.numpy as jnp
import pytest

from cosmograd import FlatLCDM
from cosmograd.background import Background
from cosmograd.testing import check_background

BACKGROUNDS: list[Background] = [FlatLCDM(h=0.67, omega_m=0.31)]


@pytest.mark.parametrize("cosmo", BACKGROUNDS, ids=lambda c: type(c).__name__)
@pytest.mark.parametrize(
    "z", [jnp.linspace(0.0, 5.0, 50), jnp.array(1.0)], ids=["vector", "scalar"]
)
def test_background_conformance(cosmo: Background, z: jnp.ndarray) -> None:
    check_background(cosmo, z)
