"""Standard cosmologies: every parameter a perturbation backend needs is set."""

import math
from typing import Any

import numpy as np
import pytest

from cosmograd import LCDM, EisensteinHu, FlatLCDM, FlatW0WaCDM, W0WaCDM, cosmologies

# sigma_8 quoted in the table each cosmology is taken from.
PUBLISHED_SIGMA8 = {"Planck13": 0.8288, "Planck15": 0.8159, "Planck18": 0.8102}


@pytest.fixture(params=list(PUBLISHED_SIGMA8))
def name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def cosmo(name: str) -> Any:
    return getattr(cosmologies, name)


@pytest.mark.parametrize("cls", [FlatLCDM, LCDM, FlatW0WaCDM, W0WaCDM])
@pytest.mark.parametrize("parameter", ["omega_b", "n_s", "ln10as"])
def test_perturbation_parameters_default_to_planck18(cls: Any, parameter: str) -> None:
    default = getattr(cls(H0=70.0, omega_m=0.3), parameter)
    assert default == getattr(cosmologies.Planck18, parameter)


def test_has_every_perturbation_parameter(cosmo: Any) -> None:
    assert np.isfinite(EisensteinHu(cosmo).matter_power_spectrum(0.1, 0.0))


@pytest.mark.slow
def test_sigma8_matches_publication(name: str, cosmo: Any) -> None:
    """A mistyped ``omega_b``, ``n_s`` or ``ln10as`` moves sigma_8 by far more."""
    camb = pytest.importorskip("camb")
    h2 = (cosmo.H0 / 100) ** 2
    # The publications assume one massive neutrino of 0.06 eV, which is part of
    # ``omega_m`` here: Omega_nu h^2 = m_nu / 93.14 eV.
    mnu = 0.06
    pars = camb.set_params(
        H0=cosmo.H0,
        ombh2=cosmo.omega_b * h2,
        omch2=(cosmo.omega_m - cosmo.omega_b) * h2 - mnu / 93.14,
        mnu=mnu,
        nnu=cosmo.n_eff,
        TCMB=cosmo.t_cmb,
        ns=cosmo.n_s,
        As=1e-10 * math.exp(cosmo.ln10as),
    )
    pars.set_matter_power(redshifts=[0.0], kmax=5.0)
    sigma8 = camb.get_results(pars).get_sigma8_0()
    np.testing.assert_allclose(sigma8, PUBLISHED_SIGMA8[name], rtol=2e-3)
