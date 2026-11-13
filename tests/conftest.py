"""Test configuration. The package itself never changes JAX precision."""

import importlib.util
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float

from cosmograd import EisensteinHu, FlatLCDM, FlatW0WaCDM, Perturbations
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


@pytest.fixture(
    params=[
        FlatLCDM(H0=67.0, omega_m=0.31),
        FlatW0WaCDM(H0=67.0, omega_m=0.31, w0=-0.9, wa=0.2),
    ],
    ids=lambda c: type(c).__name__,
)
def background(request: pytest.FixtureRequest) -> Background:
    """Every in-tree background implementation; add new backends to ``params``."""
    return request.param


class PerturbationsCase(NamedTuple):
    """A perturbations backend with inputs inside its documented range."""

    cosmo: EisensteinHu | CosmoPower | BaccoEmu
    k: Float[Array, " k"]
    """Wavenumbers in 1/Mpc, increasing."""
    z: Float[Array, " z"]
    """Redshifts, increasing, valid for the supported spectra."""
    supports_nonlinear: bool = True


@pytest.fixture(
    params=[
        pytest.param(
            PerturbationsCase(
                EisensteinHu(
                    FlatLCDM(H0=67.0, omega_m=0.31),
                    omega_b=0.049,
                    n_s=0.965,
                    ln10as=3.044,
                ),
                k=jnp.geomspace(1e-4, 1.0, 7),
                z=jnp.linspace(0.0, 10.0, 4),
                supports_nonlinear=False,
            ),
            id="eisenstein-hu",
        ),
        pytest.param(
            PerturbationsCase(
                CosmoPower(
                    FlatLCDM(H0=67.0, omega_m=0.31),
                    omega_b=0.049,
                    n_s=0.965,
                    ln10as=3.044,
                ),
                k=jnp.geomspace(1e-3, 1.0, 7),
                z=jnp.linspace(0.0, 3.0, 4),
            ),
            id="cosmopower",
            marks=_requires("cosmopower_jax"),
        ),
        pytest.param(
            PerturbationsCase(
                BaccoEmu(
                    h=0.67,
                    omega_cold=0.31,
                    omega_baryon=0.049,
                    ns=0.965,
                    sigma8_cold=0.81,
                ),
                k=jnp.geomspace(1e-3, 1.0, 7),
                z=jnp.linspace(0.0, 1.5, 4),
            ),
            id="baccoemu",
            marks=_requires("baccoemu.matter_powerspectrum"),
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
