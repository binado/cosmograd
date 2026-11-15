"""Test configuration. The package itself never changes JAX precision."""

import importlib.util
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float

from cosmograd import EisensteinHu, Perturbations
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.baccoemu._emulator import load_emulator
from cosmograd.perturbations.cosmopower import CosmoPower

from .cosmologies import EMULATOR_COSMOLOGY

jax.config.update("jax_enable_x64", True)


def pytest_configure(config: pytest.Config) -> None:
    """Fetch every baccoemu emulator once, before any test or xdist worker runs.

    The emulators cosmograd uses go through its own cache, which is safe under
    concurrency. The baryonic ones are only needed by cloelib, which loads them
    through ``baccoemu``'s downloader. That downloads into the install
    directory, racing between workers, and disables certificate checks by
    replacing ``ssl``'s default context, which ``urllib`` ignores once any other
    download, such as cloelib's CosmoPower emulators, has built its opener. So
    the downloads must come first. The variants are those cloelib loads.
    """
    if hasattr(config, "workerinput") or not _installed("baccoemu"):
        return
    from baccoemu.matter_powerspectrum import Matter_powerspectrum

    for model in ("Angulo2021", "Arico2023"):
        load_emulator(model)
    Matter_powerspectrum(baryonic_boost=False, verbose=False)
    for nonlinear in ("Angulo2021", "Arico2023"):
        for baryonic in ("Arico2021", "Burger2025"):
            Matter_powerspectrum(
                nonlinear_model_name=nonlinear,
                baryonic_model_name=baryonic,
                baryonic_boost=True,
                verbose=False,
            )


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
            marks=_requires("cosmopower_jax"),
        ),
        pytest.param(
            PerturbationsCase(
                BaccoEmu(EMULATOR_COSMOLOGY),
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
