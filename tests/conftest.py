"""Test configuration. The package itself never changes JAX precision."""

import importlib.util
import math
from dataclasses import dataclass, replace
from functools import cache
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float

from cosmograd import (
    LCDM,
    EisensteinHu,
    FlatLCDM,
    FlatW0WaCDM,
    Perturbations,
    W0WaCDM,
)
from cosmograd import cosmologies as standard_cosmologies
from cosmograd.background import Background
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

jax.config.update("jax_enable_x64", True)


def pytest_configure(config: pytest.Config) -> None:
    """Fetch the baccoemu emulators once, before any xdist worker starts.

    ``baccoemu`` downloads into its install directory on first use, deleting the
    archive after extraction, so concurrent workers would race on it.
    """
    if hasattr(config, "workerinput") or not _installed("baccoemu"):
        return
    from baccoemu.matter_powerspectrum import Matter_powerspectrum

    Matter_powerspectrum(baryonic_boost=False, verbose=False)


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


ConcreteBackground = FlatLCDM | LCDM | FlatW0WaCDM | W0WaCDM


@dataclass(frozen=True)
class CosmologyCase:
    """A background plus the primordial parameters every perturbation class needs.

    The background constants alone do not fix a perturbation calculation, so
    the baryon density and the primordial spectrum travel with them.
    """

    name: str
    background: ConcreteBackground
    omega_b: float
    """Baryon density parameter today, :math:`\\Omega_b` (not :math:`\\Omega_b h^2`)."""
    n_s: float
    ln10as: float
    """:math:`\\ln(10^{10} A_s)` at the pivot :math:`k = 0.05` 1/Mpc."""
    flat: bool = True
    lcdm: bool = True
    """Whether dark energy is a cosmological constant."""

    @property
    def cosmology(self) -> ConcreteBackground:
        """The background with the perturbation parameters set on it."""
        return self.background.with_parameters(
            omega_b=self.omega_b, n_s=self.n_s, ln10as=self.ln10as
        )

    @property
    def H0(self) -> float:
        return float(np.asarray(self.background.H0))

    @property
    def omega_m(self) -> float:
        return float(np.asarray(self.background.omega_m))

    @property
    def w0(self) -> float:
        return float(np.asarray(self.background.w0))

    @property
    def wa(self) -> float:
        return float(np.asarray(self.background.wa))


_CASES: tuple[CosmologyCase, ...] = (
    # Planck background constants, plus the baryon density and primordial
    # parameters of each release (rounded; only a fixed, plausible set of
    # inputs matters for a code-to-code comparison).
    CosmologyCase("planck18", standard_cosmologies.Planck18, 0.04897, 0.9665, 3.047),
    CosmologyCase("planck15", standard_cosmologies.Planck15, 0.0486, 0.9667, 3.064),
    CosmologyCase("planck13", standard_cosmologies.Planck13, 0.04825, 0.9611, 3.091),
    CosmologyCase(
        "planck18-w0wa",
        FlatW0WaCDM(H0=67.66, omega_m=0.3111, w0=-0.9, wa=0.2),
        0.04897,
        0.9665,
        3.047,
        lcdm=False,
    ),
    # w crosses -1 at a = 0.8, which CAMB only handles with its PPF fluid.
    CosmologyCase(
        "planck18-phantom-crossing",
        FlatW0WaCDM(H0=67.66, omega_m=0.3111, w0=-1.1, wa=0.5),
        0.04897,
        0.9665,
        3.047,
        lcdm=False,
    ),
    CosmologyCase(
        "planck18-open",
        LCDM(H0=67.66, omega_m=0.3111, omega_k=0.05),
        0.04897,
        0.9665,
        3.047,
        flat=False,
    ),
    # The general class, with every parameter free.
    CosmologyCase(
        "planck18-open-w0wa",
        W0WaCDM(H0=67.66, omega_m=0.3111, omega_k=0.05, w0=-0.9, wa=0.2),
        0.04897,
        0.9665,
        3.047,
        flat=False,
        lcdm=False,
    ),
    CosmologyCase(
        "planck18-closed",
        LCDM(H0=67.66, omega_m=0.3111, omega_k=-0.05),
        0.04897,
        0.9665,
        3.047,
        flat=False,
    ),
)


def cosmologies(
    kind: Literal["flat", "flat_lcdm"] | None = None,
) -> list[CosmologyCase]:
    """The shared list of reference cosmologies, optionally filtered.

    Parameters
    ----------
    kind : {"flat", "flat_lcdm"}, optional
        ``None`` returns every case, ``"flat"`` the flat ones (ΛCDM and w0wa)
        and ``"flat_lcdm"`` the flat ΛCDM ones.
    """
    if kind is None:
        return list(_CASES)
    if kind == "flat":
        return [c for c in _CASES if c.flat]
    if kind == "flat_lcdm":
        return [c for c in _CASES if c.flat and c.lcdm]
    raise ValueError(f"unknown kind {kind!r}")


def cosmology_params(
    kind: Literal["flat", "flat_lcdm"] | None = None,
) -> list[Any]:
    """:func:`cosmologies` as ``pytest.param`` objects with readable ids."""
    return [pytest.param(c, id=c.name) for c in cosmologies(kind)]


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
    case: CosmologyCase,
    *,
    power: bool = False,
    kmax: float = 2.0,
    zs: tuple[float, ...] = (0.0,),
) -> Any:
    """CAMB results for a case, cached because the runs are the slow part.

    Parameters
    ----------
    case : CosmologyCase
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
    bg = case.background
    key = (
        case.H0,
        case.omega_m,
        float(np.asarray(bg.omega_k)),
        case.w0,
        case.wa,
        bg.n_eff,
        bg.t_cmb,
        case.omega_b,
        case.n_s,
        case.ln10as,
    )
    return _camb_results(key, power, kmax if power else 0.0, zs if power else ())


@pytest.fixture(params=cosmology_params())
def cosmology_case(request: pytest.FixtureRequest) -> CosmologyCase:
    """Every shared reference cosmology."""
    return request.param


COSMOLOGY = FlatLCDM(H0=67.0, omega_m=0.31, omega_b=0.049, n_s=0.965, ln10as=3.044)
"""A cosmology inside every backend's supported range."""


@pytest.fixture
def background(cosmology_case: CosmologyCase) -> Background:
    """Every in-tree background implementation, on every shared cosmology."""
    return cosmology_case.background


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
                EisensteinHu(COSMOLOGY),
                k=jnp.geomspace(1e-4, 1.0, 7),
                z=jnp.linspace(0.0, 10.0, 4),
                supports_nonlinear=False,
            ),
            id="eisenstein-hu",
        ),
        pytest.param(
            PerturbationsCase(
                CosmoPower(COSMOLOGY),
                k=jnp.geomspace(1e-3, 1.0, 7),
                z=jnp.linspace(0.0, 3.0, 4),
            ),
            id="cosmopower",
            marks=_requires("cosmopower_jax"),
        ),
        pytest.param(
            PerturbationsCase(
                BaccoEmu(COSMOLOGY),
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
