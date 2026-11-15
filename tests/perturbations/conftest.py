"""Fixtures running every in-tree perturbations backend with in-range inputs."""

from dataclasses import replace
from typing import NamedTuple

import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float

from cosmograd import EisensteinHu, Perturbations
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

from ..cosmologies import EMULATOR_COSMOLOGY
from ..optional import requires


class PerturbationsCase(NamedTuple):
    """A perturbations backend with inputs inside its documented range."""

    cosmo: EisensteinHu | CosmoPower | BaccoEmu
    k: Float[Array, " k"]
    """Wavenumbers in 1/Mpc, increasing."""
    z: Float[Array, " z"]
    """Redshifts, increasing, valid for the supported spectra."""
    supports_nonlinear: bool = True
    growth_rate_rtol: float = 1e-6
    """Agreement of ``growth_rate`` with :math:`-(1 + z)\\, \\mathrm{d}\\ln D / \\mathrm{d}z`."""


@pytest.fixture(
    params=[
        pytest.param(
            PerturbationsCase(
                EisensteinHu(EMULATOR_COSMOLOGY),
                k=jnp.geomspace(1e-4, 1.0, 7),
                z=jnp.linspace(0.0, 10.0, 4),
                supports_nonlinear=False,
            ),
            id="eisenstein-hu",
        ),
        pytest.param(
            PerturbationsCase(
                CosmoPower(EMULATOR_COSMOLOGY),
                k=jnp.geomspace(1e-3, 1.0, 7),
                z=jnp.linspace(0.0, 3.0, 4),
            ),
            id="cosmopower",
            marks=requires("cosmopower_jax"),
        ),
        pytest.param(
            PerturbationsCase(
                BaccoEmu(EMULATOR_COSMOLOGY),
                k=jnp.geomspace(1e-3, 1.0, 7),
                z=jnp.linspace(0.0, 1.5, 4),
                # The emulated growth function is not smooth enough for its
                # derivative to match the growth rate more closely.
                growth_rate_rtol=5e-2,
            ),
            id="baccoemu",
            marks=requires("baccoemu.matter_powerspectrum"),
        ),
    ]
)
def perturbations_case(
    request: pytest.FixtureRequest, nonlinear: bool
) -> PerturbationsCase:
    """Every in-tree perturbations backend with in-range ``k`` and ``z``.

    Add new backends to ``params``. Emulator backends are skipped when their
    optional extra is missing.
    """
    case = request.param
    if nonlinear and not case.supports_nonlinear:
        pytest.skip("backend supports linear power only")
    return case._replace(cosmo=replace(case.cosmo, nonlinear=nonlinear))


@pytest.fixture(params=[False, True], ids=["linear", "nonlinear"])
def nonlinear(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def perturbations(perturbations_case: PerturbationsCase) -> Perturbations:
    """Every in-tree perturbations backend; see :func:`perturbations_case`."""
    return perturbations_case.cosmo
