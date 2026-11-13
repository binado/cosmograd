"""Independent growth solutions and JAX transformations for the ODE solver."""

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from cosmograd import FlatLCDM
from cosmograd.perturbations import linear_growth


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["H0", "omega_m", "w"],
    meta_fields=["t_cmb", "n_eff"],
)
@dataclass(frozen=True)
class SmoothWCDM(FlatLCDM):
    """Physical flat background with constant-w, smooth dark energy."""

    w: Real[ArrayLike, "..."] = -1.0

    def efunc(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        zp1 = 1 + jnp.asarray(z)
        return jnp.sqrt(
            self.omega_m * zp1**3
            + self.omega_r * zp1**4
            + self.omega_de * zp1 ** (3 * (1 + self.w))
        )


@pytest.fixture(params=[-1.0, -1 / 3], ids=["lambda", "curvature-like"])
def smooth_background(request: pytest.FixtureRequest) -> SmoothWCDM:
    # w=-1/3 has the same H(a) and growth as a matter-plus-curvature universe.
    return SmoothWCDM(H0=70.0, omega_m=0.3, t_cmb=0.0, w=request.param)


def test_heath_integral(smooth_background: SmoothWCDM) -> None:
    """The independent Heath integral applies for matter + Λ or curvature."""
    bg = smooth_background
    z = np.array([[3.0, 0.0], [0.5, 3.0]])
    # Reference quadrature: I(a) = integral_0^a da' / (a' E(a'))^3.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    u, weights = (nodes + 1) / 2, weights / 2
    a = 1 / (1 + z)
    ap = a[..., None] * u**2
    integral = (
        2
        * a
        * np.sum(weights * u / (ap * np.asarray(bg.efunc(1 / ap - 1))) ** 3, axis=-1)
    )
    e = np.asarray(bg.efunc(z))
    d_md = 2.5 * bg.omega_m * e * integral
    dlnh = -(
        3 * bg.omega_m * a**-3 + 3 * (1 + bg.w) * bg.omega_de * a ** (-3 * (1 + bg.w))
    ) / (2 * e**2)
    f = dlnh + 1 / (a**2 * e**3 * integral)
    d, rate = linear_growth(bg, z, a_initial=1e-5)
    np.testing.assert_allclose(d, d_md / d_md[0, 1], rtol=1e-6)
    np.testing.assert_allclose(rate, f, rtol=1e-6)
    raw, _ = linear_growth(bg, z, a_initial=1e-5, normalize=False)
    # Finite initial curvature gives an O(a_initial) amplitude error.
    np.testing.assert_allclose(raw, d_md, rtol=1e-4)


@pytest.mark.parametrize("z", [0.0, 0.5, 10.0, 999.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_einstein_de_sitter(z: float, normalize: bool) -> None:
    bg = FlatLCDM(H0=70.0, omega_m=1.0, t_cmb=0.0)
    d, f = linear_growth(bg, z, normalize=normalize)
    np.testing.assert_allclose(d, 1 / (1 + z), rtol=1e-7)
    np.testing.assert_allclose(f, 1.0, rtol=1e-7)


def test_matter_era_normalization() -> None:
    bg = FlatLCDM(H0=70.0, omega_m=0.3, t_cmb=0.0)
    d, f = linear_growth(bg, 999.0, normalize=False)
    np.testing.assert_allclose(d, 0.001)
    np.testing.assert_allclose(f, 1.0)


def test_jit_and_cosmology_vmap(smooth_background: SmoothWCDM) -> None:
    bg = smooth_background
    z = jnp.array([2.0, 0.0, 0.5])
    expected = linear_growth(bg, z)
    np.testing.assert_allclose(jax.jit(linear_growth)(bg, z), expected)
    second = replace(bg, omega_m=0.4)
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), bg, second)
    actual = jax.jit(jax.vmap(linear_growth, in_axes=(0, None)))(batch, z)
    np.testing.assert_allclose(
        actual,
        jnp.stack([jnp.array(expected), jnp.array(linear_growth(second, z))], axis=1),
    )


def test_parameter_autodiff() -> None:
    # A genuinely different expansion history exercises generic Background use.
    bg = SmoothWCDM(H0=70.0, omega_m=0.3, t_cmb=0.0, w=-0.8)
    forward = jax.jit(jax.jacfwd(lambda b: linear_growth(b, 1.0)[0]))(bg)
    reverse = jax.jit(jax.grad(lambda b: linear_growth(b, 1.0)[0]))(bg)
    np.testing.assert_allclose(
        jax.tree.leaves(forward), jax.tree.leaves(reverse), atol=1e-8
    )
    assert jnp.isfinite(forward.w) & (forward.w != 0)
    assert jnp.isfinite(forward.omega_m) & (forward.omega_m != 0)
    delta = 1e-4
    plus = linear_growth(replace(bg, w=bg.w + delta), 1.0)[0]
    minus = linear_growth(replace(bg, w=bg.w - delta), 1.0)[0]
    np.testing.assert_allclose(forward.w, (plus - minus) / (2 * delta), rtol=1e-5)


@pytest.mark.parametrize("z", [0.0, 0.5, 3.0, 10.0])
def test_growth_rate_derivative(smooth_background: SmoothWCDM, z: float) -> None:
    bg = smooth_background
    derivative = jax.grad(lambda z_: jnp.log(linear_growth(bg, z_)[0]))(z)
    _, f = linear_growth(bg, z)
    np.testing.assert_allclose(f, -(1 + z) * derivative, rtol=1e-6)


def test_default_jax_precision() -> None:
    """The library must also work with JAX's default float32 precision."""
    code = """
import jax
import numpy as np
from cosmograd import EisensteinHu, FlatLCDM
from cosmograd.perturbations import linear_growth

assert not jax.config.jax_enable_x64
bg = FlatLCDM(H0=70.0, omega_m=1.0, t_cmb=0.0)
d, f = linear_growth(bg, np.array([0.0, 1.0, 10.0]))
np.testing.assert_allclose(d, [1.0, 0.5, 1 / 11], rtol=1e-4)
np.testing.assert_allclose(f, 1.0, rtol=1e-4)
eh = EisensteinHu(FlatLCDM(H0=67.36, omega_m=0.3153), 0.0493, 0.9649, 3.044)
assert 0 < eh.growth_function(1.0) < 1
assert np.isfinite(eh.matter_power_spectrum(0.1, 1.0))
assert not jax.config.jax_enable_x64
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=os.environ | {"JAX_ENABLE_X64": "0"},
    )
