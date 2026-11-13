"""CPL background and its ΛCDM limit.

Comparisons against CAMB live in ``test_reference_background.py``."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from cosmograd import FlatW0WaCDM

H0, OMEGA_M = 67.36, 0.3153
Z = np.linspace(0.0, 10.0, 101)
RTOL = 1e-5


@pytest.fixture(
    params=[(-1.0, 0.0), (-0.8, 0.0), (-0.9, 0.3), (-1.2, -0.4)],
    ids=["lambda", "constant-w", "thawing", "phantom"],
    scope="module",
)
def cosmo(request: pytest.FixtureRequest) -> FlatW0WaCDM:
    w0, wa = request.param
    return FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=w0, wa=wa)


def test_leaves_are_parameters(cosmo: FlatW0WaCDM) -> None:
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves == [H0, OMEGA_M, cosmo.w0, cosmo.wa]


@pytest.mark.parametrize("name", ["H0", "omega_m", "w0", "wa"])
def test_gradient_matches_finite_difference(cosmo: FlatW0WaCDM, name: str) -> None:
    z = 1.0
    grad = jax.grad(lambda c: c.luminosity_distance(z))(cosmo)
    eps = 1e-6
    up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
    down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
    fd = (up.luminosity_distance(z) - down.luminosity_distance(z)) / (2 * eps)
    np.testing.assert_allclose(getattr(grad, name), fd, rtol=RTOL)
