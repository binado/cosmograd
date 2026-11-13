"""Standard cosmologies against their astropy realizations."""

from typing import Any

import astropy.cosmology
import numpy as np
import pytest

from cosmograd import EisensteinHu, cosmologies
from cosmograd.background import Background


@pytest.fixture(params=list(cosmologies.available))
def name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def cosmo(name: str) -> Background:
    return cosmologies.available[name]


@pytest.fixture
def reference(name: str) -> Any:
    return getattr(astropy.cosmology, name)


def test_baryon_density_matches_astropy(cosmo, reference) -> None:
    np.testing.assert_allclose(cosmo.omega_b, reference.Ob0)


def test_spectral_index_matches_astropy(cosmo, reference) -> None:
    np.testing.assert_allclose(cosmo.n_s, reference.meta["n"])


def test_has_every_perturbation_parameter(cosmo) -> None:
    assert np.isfinite(EisensteinHu(cosmo).matter_power_spectrum(0.1, 0.0))
