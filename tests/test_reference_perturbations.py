"""Perturbation backends against cloelib's wrappers of the same emulators.

cloelib builds each spectrum from its own background object, so these tests
catch mistakes in cosmograd's parameter mapping, units and interpolation.
BaccoEmu is the very same network on both sides. CosmoPower is not: cloelib
loads the Euclid DR1 networks (downloaded on first use) while cosmograd uses
the ``mpk_lin`` and ``mpk_boost`` networks of ``cosmopower_jax``, so that
comparison is bounded by the emulators' own accuracy.

Cases outside an emulator's training box are skipped with the reason.
"""

import math
from typing import Any

import numpy as np
import pytest

from cosmograd import FlatLCDM
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

from .conftest import CosmologyCase, cosmology_params

pytestmark = pytest.mark.slow

NONLINEAR_RTOL = 0.1
"""cloelib's nonlinear CosmoPower network includes AGN feedback (``logT_AGN`` =
7.6) that suppresses power by several per cent at k ~ 1/Mpc; cosmograd's boost
is dark-matter-only. The bound only catches gross mistakes."""

Z = np.array([0.0, 0.5, 1.0])
K = np.geomspace(1e-3, 1.0, 12)
"""Wavenumbers in 1/Mpc, inside both emulators' ranges."""


def _cloelib_background(case: CosmologyCase) -> Any:
    jax_cosmology = pytest.importorskip("cloelib.cosmology.jax_cosmology")
    return jax_cosmology.JAXBackground(
        H0=case.H0,
        Omega_b0=case.omega_b,
        Omega_cdm0=case.omega_m - case.omega_b,
        Omega_k0=0.0,
        As=math.exp(case.ln10as) * 1e-10,
        ns=case.n_s,
        mnu=0.0,
        w0=case.w0,
        wa=case.wa,
        gamma_MG=0.55,
        N_mnu=0,
    )


def _power(cosmo: Any, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Spectrum with shape ``(z, k)``, as cloelib returns it."""
    return np.asarray(cosmo.linear_matter_power_spectrum(k[None, :], z[:, None]))


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
@pytest.mark.parametrize("case", cosmology_params("flat_lcdm"))
def test_cosmopower(case: CosmologyCase, nonlinear: bool) -> None:
    module = pytest.importorskip("cloelib.cosmology.cosmopower_jax_cosmology")
    pytest.importorskip("cosmopower_jax")
    background = _cloelib_background(case)
    emu = module.CosmoPowerJAXLCDMPerturbations
    try:
        linear = emu.Linear(background, Z)
        reference = emu.NonLinear(background, linear, Z) if nonlinear else linear
    except ValueError as err:  # raised by cloelib's training-box check
        pytest.skip(str(err))
    bg = case.background
    assert isinstance(bg, FlatLCDM)
    mine = CosmoPower(
        bg,
        omega_b=case.omega_b,
        n_s=case.n_s,
        ln10as=case.ln10as,
        nonlinear=nonlinear,
    )
    expected = np.asarray(reference.matter_power_spectrum(Z, K))
    actual = (
        np.asarray(mine.matter_power_spectrum(K[None, :], Z[:, None]))
        if nonlinear
        else _power(mine, K, Z)
    )
    np.testing.assert_allclose(
        actual, expected, rtol=NONLINEAR_RTOL if nonlinear else 5e-3
    )


@pytest.mark.parametrize("case", cosmology_params("flat"))
def test_baccoemu_linear(case: CosmologyCase) -> None:
    module = pytest.importorskip("cloelib.cosmology.baccoemu_cosmology")
    pytest.importorskip("baccoemu.matter_powerspectrum")
    mine = BaccoEmu(
        h=case.H0 / 100,
        omega_cold=case.omega_m,
        omega_baryon=case.omega_b,
        ns=case.n_s,
        A_s=math.exp(case.ln10as) * 1e-10,
        w0=case.w0,
        wa=case.wa,
    )
    try:
        mine.check_bounds(K, Z)
    except ValueError as err:
        pytest.skip(str(err))
    reference = module.BACCOemuLinearPerturbations(_cloelib_background(case), Z)
    np.testing.assert_allclose(
        _power(mine, K, Z), np.asarray(reference.matter_power_spectrum(Z, K)), rtol=5e-3
    )


@pytest.mark.parametrize("case", cosmology_params("flat"))
def test_baccoemu_nonlinear(case: CosmologyCase) -> None:
    module = pytest.importorskip("cloelib.cosmology.baccoemu_cosmology")
    pytest.importorskip("baccoemu.matter_powerspectrum")
    mine = BaccoEmu(
        h=case.H0 / 100,
        omega_cold=case.omega_m,
        omega_baryon=case.omega_b,
        ns=case.n_s,
        A_s=math.exp(case.ln10as) * 1e-10,
        w0=case.w0,
        wa=case.wa,
        nonlinear=True,
    )
    try:
        mine.check_bounds(K, Z)
    except ValueError as err:
        pytest.skip(str(err))
    background = _cloelib_background(case)
    linear = module.BACCOemuLinearPerturbations(background, Z)
    reference = module.BACCOemuNonLinearPerturbations(background, linear, Z)
    np.testing.assert_allclose(
        np.asarray(mine.matter_power_spectrum(K[None, :], Z[:, None])),
        np.asarray(reference.matter_power_spectrum(Z, K)),
        # cloelib interpolates P linearly in k on the coarse nonlinear grid,
        # cosmograd in ln k; they differ by up to ~2% between nodes.
        rtol=2e-2,
    )
