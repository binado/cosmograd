from collections.abc import Callable
from dataclasses import replace

import jax.numpy as jnp
import pytest
from jax import Array

from cosmograd.background import Background
from cosmograd.testing import check_background, check_perturbations

from .conftest import PerturbationsCase


@pytest.mark.parametrize(
    "z", [jnp.linspace(0.0, 5.0, 50), jnp.array(1.0)], ids=["vector", "scalar"]
)
def test_background_conformance(background: Background, z: jnp.ndarray) -> None:
    check_background(background, z)


def test_perturbations_background_conformance(
    perturbations_case: PerturbationsCase,
) -> None:
    check_background(perturbations_case.cosmo.background, perturbations_case.z)


Layout = Callable[[Array, Array], tuple[Array, Array]]


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(lambda k, z: (k[:, None], z[None, :]), id="outer"),
        pytest.param(lambda k, z: (k[: z.size], z), id="elementwise"),
        pytest.param(lambda k, z: (k[k.size // 2], z[-1]), id="scalars"),
        pytest.param(lambda k, z: (k, z[-1]), id="k-vector"),
    ],
)
def test_perturbations_conformance(
    perturbations_case: PerturbationsCase, layout: Layout
) -> None:
    k, z = layout(perturbations_case.k, perturbations_case.z)
    check_perturbations(
        perturbations_case.cosmo,
        k,
        z,
        linear_cosmo=replace(perturbations_case.cosmo, nonlinear=False),
    )
