"""Backgrounds against independent references.

The closed-form universes (Einstein-de Sitter, Milne, de Sitter) run in the fast
suite and are exact. CAMB runs over every shared cosmology and is slow. CAMB is
run with ``mnu = 0`` (cosmograd folds massive neutrinos into ``omega_m``) and
PPF dark energy for the w0wa models. See :mod:`tests.background.reference`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _pytest.mark.structures import ParameterSet

from cosmograd import LCDM, FlatLCDM

from ..cosmologies import COSMOLOGIES
from .reference import CAMBBackground, DeSitter, EinsteinDeSitter, Milne

# Direct evaluations agree to ~1e-9. The inversion is a root solve (3e-8) and
# omega_r is limited by the digits kept in the photon constant (1e-8).
RTOL = 1e-8
RTOL_LOOSE = 1e-7
Z = np.concatenate([np.linspace(0.01, 10.0, 60), [30.0, 100.0, 500.0, 1200.0]])
METHODS = [
    "hubble_parameter",
    "matter_density_parameter",
    "comoving_distance",
    "transverse_comoving_distance",
    "luminosity_distance",
    "angular_diameter_distance",
    "differential_comoving_volume",
]
H0 = 70.0


@dataclass(frozen=True)
class Pair:
    """A cosmograd background and an independent reference of the same universe."""

    ours: Any
    reference: Any
    z_inverse_max: float = 1200.0
    """Largest redshift the inverse distance relation is tested at."""


def _analytic(
    name: str, ours: Any, make: Callable[[], Any], **kw: float
) -> ParameterSet:
    return pytest.param(lambda: Pair(ours, make(), **kw), id=name)


ANALYTIC: list[ParameterSet] = [
    _analytic(
        "einstein-de-sitter",
        FlatLCDM(H0=H0, omega_m=1.0, t_cmb=0.0),
        lambda: EinsteinDeSitter(H0=H0),
    ),
    _analytic(
        "milne",
        LCDM(H0=H0, omega_m=0.0, omega_k=1.0, t_cmb=0.0),
        lambda: Milne(H0=H0),
    ),
    # The fixed number of Newton iterations in the inverse relation converges
    # to z ~ 4 here, since chi grows linearly in z, and diverges beyond z ~ 7.
    _analytic(
        "de-sitter",
        FlatLCDM(H0=H0, omega_m=0.0, t_cmb=0.0),
        lambda: DeSitter(H0=H0),
        z_inverse_max=3.0,
    ),
]
CAMB: list[ParameterSet] = [
    pytest.param(
        lambda cosmo=cosmo: Pair(cosmo, CAMBBackground.from_background(cosmo)),
        id=f"camb-{name}",
        marks=pytest.mark.slow,
    )
    for name, cosmo in COSMOLOGIES.items()
]


def _make(request: pytest.FixtureRequest) -> Pair:
    return request.param()


@pytest.fixture(params=ANALYTIC + CAMB)
def pair(request: pytest.FixtureRequest) -> Pair:
    return _make(request)


@pytest.fixture(params=ANALYTIC)
def analytic_pair(request: pytest.FixtureRequest) -> Pair:
    return _make(request)


@pytest.fixture(params=CAMB)
def camb_pair(request: pytest.FixtureRequest) -> Pair:
    return _make(request)


@pytest.mark.parametrize("method", METHODS)
def test_method_matches_reference(pair: Pair, method: str) -> None:
    np.testing.assert_allclose(
        getattr(pair.ours, method)(Z), getattr(pair.reference, method)(Z), rtol=RTOL
    )


def test_redshift_at_comoving_distance_matches_reference(pair: Pair) -> None:
    z = Z[Z <= pair.z_inverse_max]
    chi = jnp.asarray(pair.reference.comoving_distance(z))
    np.testing.assert_allclose(
        pair.ours.redshift_at_comoving_distance(chi), z, rtol=RTOL_LOOSE
    )


def test_radiation_density_matches_camb(camb_pair: Pair) -> None:
    np.testing.assert_allclose(
        camb_pair.ours.omega_r, camb_pair.reference.omega_r, rtol=RTOL_LOOSE
    )


@pytest.mark.parametrize("method", METHODS)
def test_hubble_constant_gradient_matches_closed_form(
    analytic_pair: Pair, method: str
) -> None:
    def jacobian(cosmo: Any) -> jax.Array:
        return jax.jacfwd(lambda c: getattr(c, method)(Z))(cosmo).H0

    np.testing.assert_allclose(
        jacobian(analytic_pair.ours), jacobian(analytic_pair.reference), rtol=RTOL
    )


def test_inverse_distance_gradient_matches_closed_form(analytic_pair: Pair) -> None:
    """Differentiating through the root solve gives the exact derivative."""
    z = Z[(Z > 0.1) & (Z <= analytic_pair.z_inverse_max)]
    chi = jnp.asarray(analytic_pair.reference.comoving_distance(z))

    def jacobian(cosmo: Any) -> jax.Array:
        return jax.jacfwd(lambda c: c.redshift_at_comoving_distance(chi))(cosmo).H0

    np.testing.assert_allclose(
        jacobian(analytic_pair.ours), jacobian(analytic_pair.reference), rtol=RTOL_LOOSE
    )
