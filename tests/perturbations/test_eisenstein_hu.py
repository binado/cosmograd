"""EH98 physical identities and native JAX transformation behaviour."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import EisensteinHu, FlatLCDM, Perturbations
from cosmograd.perturbations import EisensteinHu as ExportedEisensteinHu

PLANCK = FlatLCDM(H0=67.36, omega_m=0.3153, omega_b=0.0493, n_s=0.9649, ln10as=3.044)


@pytest.fixture(
    params=[
        PLANCK,
        FlatLCDM(H0=72.0, omega_m=0.25, omega_b=0.04, n_s=1.0, ln10as=3.1),
        FlatLCDM(H0=62.0, omega_m=0.4, omega_b=0.07, n_s=0.94, ln10as=2.9, t_cmb=2.8),
    ],
    ids=["planck", "low-matter", "high-matter"],
)
def eh(request: pytest.FixtureRequest) -> EisensteinHu:
    return EisensteinHu(request.param)


def test_background_and_pytree(eh: EisensteinHu) -> None:
    bg = eh.background
    cosmo = EisensteinHu(bg)
    assert cosmo.background is bg
    assert isinstance(cosmo, Perturbations)
    assert ExportedEisensteinHu is EisensteinHu
    np.testing.assert_array_equal(
        jax.tree_util.tree_leaves(cosmo),
        [bg.H0, bg.omega_m, bg.omega_b, bg.n_s, bg.ln10as],
    )
    with pytest.raises(FrozenInstanceError):
        setattr(cosmo, "nonlinear", True)


def test_nonlinear_rejected(eh: EisensteinHu) -> None:
    with pytest.raises(NotImplementedError, match="linear power only"):
        replace(eh, nonlinear=True)


def test_growth_is_independent_of_wavenumber(eh: EisensteinHu) -> None:
    k = jnp.array([0.2, 0.001, 0.05])
    z = jnp.array([10.0, 0.0, 1.0])
    np.testing.assert_allclose(
        eh.growth_function(z[:, None], k=k),
        np.broadcast_to(eh.growth_function(z)[:, None], (3, 3)),
    )
    np.testing.assert_allclose(
        eh.growth_rate(z[:, None], k=k),
        np.broadcast_to(eh.growth_rate(z)[:, None], (3, 3)),
    )
