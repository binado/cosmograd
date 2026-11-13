"""Curved and general backgrounds and their constrained limits.

Comparisons against CAMB live in ``test_reference_background.py``."""

from dataclasses import replace
from typing import Any

import jax
import numpy as np
import pytest

from cosmograd import LCDM, FlatLCDM, FlatW0WaCDM, W0WaCDM

H0, OMEGA_M = 67.36, 0.3153
Z = np.linspace(0.0, 10.0, 101)
RTOL = 1e-5


@pytest.fixture(
    params=[
        (-0.1, -1.0, 0.0),
        (-0.02, -1.0, 0.0),
        (0.05, -1.0, 0.0),
        (0.1, -0.9, 0.3),
        (-0.05, -1.2, -0.4),
    ],
    ids=["closed", "nearly-flat-closed", "open", "open-thawing", "closed-phantom"],
)
def cosmo(request: pytest.FixtureRequest) -> W0WaCDM:
    omega_k, w0, wa = request.param
    return W0WaCDM(H0=H0, omega_m=OMEGA_M, omega_k=omega_k, w0=w0, wa=wa)


@pytest.mark.parametrize("omega_k", [0.0, 1e-8, -1e-8, 1e-3])
def test_w0wacdm_curvature_gradient_is_finite_near_flat(omega_k: float) -> None:
    cosmo = W0WaCDM(H0=H0, omega_m=OMEGA_M, omega_k=omega_k)
    grad = jax.grad(lambda c: c.luminosity_distance(1.0))(cosmo)
    assert np.isfinite(grad.omega_k)


@pytest.mark.parametrize("name", ["H0", "omega_m", "omega_k", "w0", "wa"])
def test_w0wacdm_gradient_matches_finite_difference(cosmo: W0WaCDM, name: str) -> None:
    z = 1.0
    grad = jax.grad(lambda c: c.luminosity_distance(z))(cosmo)
    eps = 1e-6
    up = replace(cosmo, **{name: getattr(cosmo, name) + eps})
    down = replace(cosmo, **{name: getattr(cosmo, name) - eps})
    fd = (up.luminosity_distance(z) - down.luminosity_distance(z)) / (2 * eps)
    np.testing.assert_allclose(getattr(grad, name), fd, rtol=RTOL)


def test_w0wacdm_leaves_are_parameters(cosmo: W0WaCDM) -> None:
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves == [
        H0,
        OMEGA_M,
        cosmo.omega_k,
        cosmo.w0,
        cosmo.wa,
        cosmo.omega_b,
        cosmo.n_s,
        cosmo.ln10as,
    ]


@pytest.mark.parametrize(
    ("constrained", "leaves"),
    [
        (FlatLCDM(H0=H0, omega_m=OMEGA_M), [H0, OMEGA_M]),
        (LCDM(H0=H0, omega_m=OMEGA_M, omega_k=0.05), [H0, OMEGA_M, 0.05]),
        (
            FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=-0.9, wa=0.3),
            [H0, OMEGA_M, -0.9, 0.3],
        ),
    ],
    ids=["flat-lcdm", "lcdm", "flat-w0wacdm"],
)
def test_constrained_class_leaves_exclude_fixed_parameters(
    constrained: Any, leaves: list[float]
) -> None:
    perturbation = [constrained.omega_b, constrained.n_s, constrained.ln10as]
    assert jax.tree_util.tree_leaves(constrained) == leaves + perturbation


@pytest.mark.parametrize(
    ("constrained", "fixed"),
    [
        (FlatLCDM(H0=H0, omega_m=OMEGA_M), {"omega_k": 0.0, "w0": -1.0, "wa": 0.0}),
        (FlatW0WaCDM(H0=H0, omega_m=OMEGA_M), {"omega_k": 0.0}),
        (LCDM(H0=H0, omega_m=OMEGA_M), {"w0": -1.0, "wa": 0.0}),
    ],
    ids=["flat-lcdm", "flat-w0wacdm", "lcdm"],
)
def test_constrained_class_fixed_parameters_have_expected_values(
    constrained: Any, fixed: dict[str, float]
) -> None:
    assert {name: getattr(constrained, name) for name in fixed} == fixed
