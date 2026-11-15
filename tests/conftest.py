"""Test configuration. The package itself never changes JAX precision."""

import importlib.util
import math
from dataclasses import replace
from functools import cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float

from cosmograd import EisensteinHu, Perturbations
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.baccoemu._emulator import load_emulator
from cosmograd.perturbations.cosmopower import CosmoPower

from .cosmologies import EMULATOR_COSMOLOGY, Cosmology

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


@cache
def _camb_results(
    key: tuple[float, ...], power: bool, kmax: float, zs: tuple[float, ...]
) -> Any:
    camb = pytest.importorskip("camb")
    h0, omega_m, omega_k, w0, wa, n_eff, t_cmb, omega_b, n_s, ln10as = key
    h2 = (h0 / 100) ** 2
    # cosmograd treats massive neutrinos as matter, so match with mnu = 0 and
    # put the whole of omega_m into baryons plus CDM.
    pars = camb.set_params(
        H0=h0,
        ombh2=omega_b * h2,
        omch2=(omega_m - omega_b) * h2,
        omk=omega_k,
        mnu=0.0,
        nnu=n_eff,
        TCMB=t_cmb,
        ns=n_s,
        As=math.exp(ln10as) * 1e-10,
        kmax=kmax,
    )
    if (w0, wa) != (-1.0, 0.0):
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
    if not power:
        return camb.get_background(pars)
    pars.set_matter_power(redshifts=list(zs), kmax=kmax)
    return camb.get_results(pars)


def camb_results(
    cosmo: Cosmology,
    *,
    power: bool = False,
    kmax: float = 2.0,
    zs: tuple[float, ...] = (0.0,),
) -> Any:
    """CAMB results for a cosmology, cached because the runs are the slow part.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology to compute.
    power : bool, default False
        Also compute the linear matter power spectrum. Without it only the
        background is solved.
    kmax : float, default 2.0
        Largest wavenumber in 1/Mpc, when ``power`` is set.
    zs : tuple of float, default (0.0,)
        Redshifts at which power is stored, when ``power`` is set.

    Returns
    -------
    camb.results.CAMBdata
        Skips the test when CAMB is not installed.

    Notes
    -----
    ``omega_m`` in cosmograd folds massive neutrinos into matter, so CAMB is
    run with ``mnu = 0`` and ``omch2 = (omega_m - omega_b) h^2``.
    """
    key = tuple(
        float(np.asarray(getattr(cosmo, name)))
        for name in (
            "H0",
            "omega_m",
            "omega_k",
            "w0",
            "wa",
            "n_eff",
            "t_cmb",
            "omega_b",
            "n_s",
            "ln10as",
        )
    )
    return _camb_results(key, power, kmax if power else 0.0, zs if power else ())


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
