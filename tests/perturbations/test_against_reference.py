"""Perturbation backends against independent implementations.

Each scenario pairs a backend with a reference and the cosmologies both support,
and compares the matter power spectrum on a ``(k, z)`` grid. The references
build each spectrum from their own parameter mapping, so these tests catch
mistakes in cosmograd's parameter mapping, units and interpolation. See
:mod:`tests.perturbations.reference`.

BaccoEmu is the very same network on both sides. CosmoPower is not: cloelib
loads the Euclid DR1 networks (downloaded on first use) while cosmograd uses the
``mpk_lin`` and ``mpk_boost`` networks of ``cosmopower_jax``, so that comparison
is bounded by the emulators' own accuracy.

Cases outside an emulator's training box are skipped with the reason.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _pytest.mark.structures import ParameterSet
from jax import Array

from cosmograd import EisensteinHu, FlatLCDM, Perturbations
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

from ..cosmologies import Cosmology, cosmology_params
from .reference import (
    CAMBPerturbations,
    CloelibPerturbations,
    JaxCosmoPerturbations,
    sigma8,
)

pytestmark = pytest.mark.slow

CLOELIB_NONLINEAR_COSMOPOWER_RTOL = 0.1
"""cloelib's nonlinear CosmoPower network includes AGN feedback (``logT_AGN`` =
7.6) that suppresses power by several per cent at k ~ 1/Mpc; cosmograd's boost
is dark-matter-only. The bound only catches gross mistakes."""

CLOELIB_NONLINEAR_BACCOEMU_RTOL = 2e-2
"""cloelib interpolates P linearly in k on the coarse nonlinear grid, cosmograd
in ln k; they differ by up to ~2% between nodes."""

EH_CAMB_RTOL = 8e-2
"""EH98 is a fitting formula and cannot match a Boltzmann code more closely than
a few per cent (the BAO phase shifts it most); it is a sanity bound on shape and
amplitude."""


@dataclass(frozen=True)
class Scenario:
    """A backend, the reference it is compared with, and where both are valid."""

    backend: Callable[..., Perturbations]
    reference: Any
    kind: Literal["flat", "flat_lcdm"]
    """The cosmologies both support."""
    rtol: float
    z: np.ndarray
    k: np.ndarray
    """Wavenumbers in 1/Mpc, inside the supported range of both."""


K_COARSE = np.geomspace(1e-3, 1.0, 12)
Z_EMULATOR = np.array([0.0, 0.5, 1.0])
SCENARIOS = {
    "eisenstein-hu-vs-jax-cosmo": Scenario(
        EisensteinHu,
        JaxCosmoPerturbations,
        "flat",
        1e-2,
        np.array([0.0, 1.0, 2.0]),
        K_COARSE,
    ),
    "eisenstein-hu-vs-camb": Scenario(
        EisensteinHu,
        CAMBPerturbations,
        "flat",
        EH_CAMB_RTOL,
        np.array([0.0, 1.0]),
        np.geomspace(1e-3, 1.0, 40),
    ),
    "cosmopower-vs-cloelib-linear": Scenario(
        CosmoPower, CloelibPerturbations, "flat_lcdm", 5e-3, Z_EMULATOR, K_COARSE
    ),
    "cosmopower-vs-cloelib-nonlinear": Scenario(
        partial(CosmoPower, nonlinear=True),
        CloelibPerturbations,
        "flat_lcdm",
        CLOELIB_NONLINEAR_COSMOPOWER_RTOL,
        Z_EMULATOR,
        K_COARSE,
    ),
    "baccoemu-vs-cloelib-linear": Scenario(
        BaccoEmu, CloelibPerturbations, "flat", 5e-3, Z_EMULATOR, K_COARSE
    ),
    "baccoemu-vs-cloelib-nonlinear": Scenario(
        partial(BaccoEmu, nonlinear=True),
        CloelibPerturbations,
        "flat",
        CLOELIB_NONLINEAR_BACCOEMU_RTOL,
        Z_EMULATOR,
        K_COARSE,
    ),
}


def _cases() -> list[ParameterSet]:
    return [
        pytest.param(scenario, cosmo.values[0], id=f"{name}-{cosmo.id}")
        for name, scenario in SCENARIOS.items()
        for cosmo in cosmology_params(scenario.kind)
    ]


@pytest.mark.parametrize(("scenario", "cosmo"), _cases())
def test_matter_power_spectrum_matches_reference(
    scenario: Scenario, cosmo: Cosmology
) -> None:
    try:
        ours = scenario.backend(cosmo)
        if isinstance(ours, BaccoEmu):
            ours.check_bounds(scenario.k, scenario.z)
        reference = scenario.reference.from_perturbations(ours, scenario.z)
    except ValueError as err:  # raised by an emulator's training-box check
        pytest.skip(str(err))
    k, z = scenario.k[:, None], scenario.z[None, :]
    np.testing.assert_allclose(
        ours.matter_power_spectrum(k, z),
        reference.matter_power_spectrum(k, z),
        rtol=scenario.rtol,
    )


@pytest.mark.parametrize("name", ["omega_m", "omega_b"])
@pytest.mark.parametrize("case", cosmology_params("flat_lcdm"))
def test_shape_gradient_matches_jax_cosmo(case: FlatLCDM, name: str) -> None:
    # The ratio P(k) / P(k_ref) is independent of the amplitude, so each code
    # can be differentiated without coupling sigma8 to the parameters.
    k = np.array([0.02, 0.1, 0.5])
    amplitude = sigma8(EisensteinHu(case))

    def ours(omega_m: Array, omega_b: Array) -> Array:
        cosmo = replace(case, omega_m=omega_m, omega_b=omega_b)
        return jnp.log(EisensteinHu(cosmo).linear_matter_power_spectrum(k, 0.0))

    def theirs(omega_m: Array, omega_b: Array) -> Array:
        cosmo = replace(case, omega_m=omega_m, omega_b=omega_b)
        return jnp.log(
            JaxCosmoPerturbations(cosmo, amplitude).linear_matter_power_spectrum(k, 0.0)
        )

    argnum = ("omega_m", "omega_b").index(name)
    point = (jnp.asarray(case.omega_m), jnp.asarray(case.omega_b))
    ours_jac = jax.jacfwd(ours, argnums=argnum)(*point)
    theirs_jac = jax.jacfwd(theirs, argnums=argnum)(*point)
    np.testing.assert_allclose(
        jnp.diff(ours_jac), jnp.diff(theirs_jac), rtol=5e-2, atol=1e-2
    )
