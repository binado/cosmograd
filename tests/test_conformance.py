import jax.numpy as jnp
import pytest
from jax import Array

from cosmograd import Perturbations
from cosmograd.background import Background
from cosmograd.testing import check_background, check_perturbations


@pytest.mark.parametrize(
    "z", [jnp.linspace(0.0, 5.0, 50), jnp.array(1.0)], ids=["vector", "scalar"]
)
def test_background_conformance(background: Background, z: jnp.ndarray) -> None:
    check_background(background, z)


def test_perturbations_background_conformance(perturbations: Perturbations) -> None:
    check_background(perturbations.background, jnp.linspace(0.0, 5.0, 50))


@pytest.mark.parametrize(
    "layout", ["outer", "elementwise", "scalars", "k-vector"], ids=str
)
def test_perturbations_conformance(
    perturbations: Perturbations,
    perturbations_grid: tuple[Array, Array],
    layout: str,
) -> None:
    k, z = perturbations_grid
    k, z = {
        "outer": (k[:, None], z[None, :]),
        "elementwise": (k[: z.size], z),
        "scalars": (k[k.size // 2], z[-1]),
        "k-vector": (k, z[-1]),
    }[layout]
    check_perturbations(perturbations, k, z)
