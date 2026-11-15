"""Physics checks specific to the CosmoPower backend."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import FlatLCDM
from cosmograd.perturbations.cosmopower import CosmoPower

pytest.importorskip("cosmopower_jax")


@pytest.fixture
def cosmo() -> CosmoPower:
    return CosmoPower(
        FlatLCDM(H0=67.4, omega_m=0.315, omega_b=0.0493, n_s=0.965, ln10as=3.044)
    )


def test_background_and_pytree(cosmo: CosmoPower) -> None:
    bg = replace(cosmo.background, t_cmb=2.8)
    cosmo = replace(cosmo, background=bg)
    assert cosmo.background is bg
    np.testing.assert_array_equal(
        jax.tree_util.tree_leaves(cosmo),
        [
            bg.H0,
            bg.omega_m,
            bg.omega_b,
            bg.n_s,
            bg.ln10as,
            cosmo.c_min,
            cosmo.eta_0,
        ],
    )


@pytest.mark.parametrize("k", [1e-2, 1e-1, 1.0])
def test_growth_is_scale_independent(cosmo: CosmoPower, k: float) -> None:
    z = jnp.linspace(0.0, 4.0, 9)
    np.testing.assert_allclose(
        cosmo.growth_function(z, k=k), cosmo.growth_function(z), rtol=2e-3
    )
