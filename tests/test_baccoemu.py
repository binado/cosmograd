"""The baccoemu backend against upstream ``baccoemu``. Skipped without the extra."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import FlatLCDM, FlatW0WaCDM, Perturbations
from cosmograd.perturbations import linear_growth
from cosmograd.perturbations.baccoemu import BaccoEmu

matter_powerspectrum = pytest.importorskip("baccoemu.matter_powerspectrum")

RTOL = 1e-5
# h/Mpc, inside both the linear range and the Angulo2021 nonlinear range, and
# spanning the linear-only regime below the nonlinear grid.
K_H = np.geomspace(1e-3, 4.5, 120)
Z = np.array([0.0, 0.4, 1.0, 1.5])
COSMOLOGY_PARAMETERS = ("H0", "omega_m", "w0", "wa", "omega_b", "n_s")
BACKEND_PARAMETERS = ("sigma8_cold", "neutrino_mass")

# Keyword arguments of ``baccoemu``, the reference for each case.
BACCO = {
    "lcdm": {
        "hubble": 0.67,
        "omega_cold": 0.315,
        "omega_baryon": 0.05,
        "ns": 0.96,
        "sigma8_cold": 0.8,
        "neutrino_mass": 0.0,
        "w0": -1.0,
        "wa": 0.0,
    },
    "nu_w0wa": {
        "hubble": 0.72,
        "omega_cold": 0.27,
        "omega_baryon": 0.045,
        "ns": 0.94,
        "sigma8_cold": 0.75,
        "neutrino_mass": 0.15,
        "w0": -0.95,
        "wa": 0.1,
    },
    "heavy_nu": {
        "hubble": 0.62,
        "omega_cold": 0.36,
        "omega_baryon": 0.055,
        "ns": 1.0,
        "sigma8_cold": 0.85,
        "neutrino_mass": 0.3,
        "w0": -1.1,
        "wa": -0.2,
    },
}


def backend(p: dict[str, float], **config: Any) -> BaccoEmu:
    """The backend whose emulator inputs are the ``baccoemu`` arguments ``p``."""
    omega_nu = p["neutrino_mass"] / 93.14 / p["hubble"] ** 2
    cosmology = FlatW0WaCDM(
        H0=100 * p["hubble"],
        omega_m=p["omega_cold"] + omega_nu,
        w0=p["w0"],
        wa=p["wa"],
        omega_b=p["omega_baryon"],
        n_s=p["ns"],
    )
    return BaccoEmu(
        cosmology,
        sigma8_cold=p["sigma8_cold"],
        neutrino_mass=p["neutrino_mass"],
        **config,
    )


def little_h(cosmo: BaccoEmu) -> Any:
    return cosmo.background.H0 / 100


LCDM, NU_W0WA, HEAVY_NU = (backend(p) for p in BACCO.values())


@pytest.fixture(scope="module")
def reference() -> Any:
    return matter_powerspectrum.Matter_powerspectrum(
        baryonic_boost=False, verbose=False
    )


@pytest.fixture(params=list(BACCO))
def bacco_params(request: pytest.FixtureRequest) -> dict[str, float]:
    return BACCO[request.param]


@pytest.fixture
def params(bacco_params: dict[str, float]) -> BaccoEmu:
    return backend(bacco_params)


@pytest.fixture(params=[False, True], ids=["total", "cold"])
def cosmo(request: pytest.FixtureRequest, params: BaccoEmu) -> BaccoEmu:
    return replace(params, cold=request.param)


def bacco_kwargs(bacco_params: dict[str, float], z: float) -> dict[str, float]:
    """Arguments of ``Matter_powerspectrum`` methods at ``z``."""
    return bacco_params | {"expfactor": 1 / (1 + z)}


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_power_spectrum_matches_baccoemu(
    cosmo: BaccoEmu, bacco_params: dict[str, float], reference: Any, nonlinear: bool
) -> None:
    cosmo = replace(cosmo, nonlinear=nonlinear)
    method = reference.get_nonlinear_pk if nonlinear else reference.get_linear_pk
    expected = [
        method(k=K_H, cold=cosmo.cold, **bacco_kwargs(bacco_params, z))[1] for z in Z
    ]
    h = bacco_params["hubble"]
    pk = cosmo.matter_power_spectrum(K_H * h, Z[:, None])
    np.testing.assert_allclose(pk * h**3, np.array(expected), rtol=RTOL)


def test_ln10as_input_matches_sigma8_input(
    params: BaccoEmu, bacco_params: dict[str, float], reference: Any
) -> None:
    a_s = reference.get_A_s(**bacco_kwargs(bacco_params, 0.0))
    cosmology = replace(params.background, ln10as=np.log(1e10 * a_s))
    from_a_s = replace(params, background=cosmology, sigma8_cold=None)
    k = K_H * bacco_params["hubble"]
    np.testing.assert_allclose(
        replace(from_a_s, nonlinear=True).matter_power_spectrum(k, 0.5),
        replace(params, nonlinear=True).matter_power_spectrum(k, 0.5),
        rtol=RTOL,
    )


def test_primordial_power_spectrum_at_pivot_is_A_s(
    params: BaccoEmu, bacco_params: dict[str, float], reference: Any
) -> None:
    a_s = reference.get_A_s(**bacco_kwargs(bacco_params, 0.0))
    np.testing.assert_allclose(params.primordial_power_spectrum(0.05), a_s, rtol=RTOL)


def test_growth_function_matches_baccoemu(
    cosmo: BaccoEmu, bacco_params: dict[str, float], reference: Any
) -> None:
    pk = np.array(
        [
            reference.get_linear_pk(
                k=K_H, cold=cosmo.cold, **bacco_kwargs(bacco_params, z)
            )[1]
            for z in Z
        ]
    )
    growth = cosmo.growth_function(Z[:, None], k=K_H * bacco_params["hubble"])
    np.testing.assert_allclose(growth, np.sqrt(pk / pk[0]), rtol=RTOL)


def test_growth_rate_lcdm_matches_omega_m_power_law() -> None:
    """Against the fit Omega_m(z)^0.55, itself good to a few 1e-3 in LCDM."""
    z = np.linspace(0.0, 3.0, 61)
    expected = LCDM.background.matter_density_parameter(z) ** 0.55
    np.testing.assert_allclose(LCDM.growth_rate(z), expected, atol=0.02)


def test_growth_function_massive_nu_is_scale_dependent() -> None:
    def scale_dependence(cosmo: BaccoEmu) -> jax.Array:
        growth = cosmo.growth_function(3.0, k=np.array([1e-3, 1.0]) * little_h(cosmo))
        return growth[1] / growth[0] - 1

    # Neutrinos slow small-scale growth, so normalised to z = 0 the
    # small-scale D is larger at high redshift.
    assert scale_dependence(HEAVY_NU) > 10 * abs(scale_dependence(LCDM))


def test_nonlinear_power_spectrum_on_large_scales_is_linear(cosmo: BaccoEmu) -> None:
    k = np.geomspace(1e-4, 5e-3, 10) * little_h(cosmo)
    np.testing.assert_allclose(
        replace(cosmo, nonlinear=True).matter_power_spectrum(k, 1.0),
        cosmo.matter_power_spectrum(k, 1.0),
    )


def _grid(shape: tuple[int, ...], lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, int(np.prod(shape))).reshape(shape)


@pytest.mark.parametrize(
    ("k_shape", "z_shape"),
    [((), ()), ((7,), ()), ((), (3,)), ((7,), (7,)), ((7, 1), (3,)), ((1, 7), (3, 1))],
)
@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_power_spectrum_broadcasts_elementwise(
    k_shape: tuple[int, ...], z_shape: tuple[int, ...], nonlinear: bool
) -> None:
    k, z = _grid(k_shape, 0.01, 2.0), _grid(z_shape, 0.2, 1.2)
    cosmo = replace(NU_W0WA, nonlinear=nonlinear)
    pk = cosmo.matter_power_spectrum(k, z)
    k_b, z_b = np.broadcast_arrays(k, z)
    expected = [
        cosmo.matter_power_spectrum(ki, zi)
        for ki, zi in zip(k_b.ravel(), z_b.ravel(), strict=True)
    ]
    assert pk.shape == k_b.shape
    np.testing.assert_allclose(pk, np.reshape(expected, k_b.shape), rtol=1e-12)


@pytest.mark.parametrize("method", ["growth_function", "growth_rate"])
@pytest.mark.parametrize(
    ("k", "shape"), [(None, (4,)), (np.full((5, 1), 0.1), (5, 4))], ids=["none", "k"]
)
def test_growth_has_broadcast_shape(
    method: str, k: np.ndarray | None, shape: tuple[int, ...]
) -> None:
    z = np.linspace(0.0, 1.0, 4)
    assert getattr(NU_W0WA, method)(z, k=k).shape == shape


def test_background_is_the_given_cosmology() -> None:
    cosmology = FlatLCDM(H0=67.0, omega_m=0.315, omega_b=0.05, n_s=0.96, ln10as=3.0)
    cosmo = BaccoEmu(cosmology)
    assert isinstance(cosmo, Perturbations)
    assert cosmo.background is cosmology


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_flat_lcdm_matches_default_flat_w0wa_cdm(nonlinear: bool) -> None:
    w0wa = replace(LCDM, nonlinear=nonlinear)
    bg = w0wa.background
    lcdm = replace(
        w0wa,
        background=FlatLCDM(
            H0=bg.H0, omega_m=bg.omega_m, omega_b=bg.omega_b, n_s=bg.n_s
        ),
    )
    k = K_H * little_h(w0wa)
    np.testing.assert_allclose(
        lcdm.matter_power_spectrum(k, 0.5), w0wa.matter_power_spectrum(k, 0.5)
    )


@pytest.mark.parametrize(("w0", "wa"), [(-1.0, 0.0), (-0.8, 0.0), (-1.1, 0.3)])
def test_growth_function_matches_ode_on_background(w0: float, wa: float) -> None:
    cosmo = backend(BACCO["lcdm"] | {"w0": w0, "wa": wa})
    z = np.linspace(0.0, 3.0, 7)
    # The emulated growth is matter-era growth, so drop radiation from the ODE.
    expected, _ = linear_growth(replace(cosmo.background, t_cmb=0.0), z)
    # Well inside the horizon, where dark energy perturbations do not matter.
    growth = cosmo.growth_function(z, k=0.1)
    np.testing.assert_allclose(growth, expected, rtol=5e-3)


def test_pytree_leaves_are_the_given_parameters() -> None:
    leaves = jax.tree_util.tree_leaves(NU_W0WA)
    expected = [getattr(NU_W0WA.background, p) for p in COSMOLOGY_PARAMETERS]
    expected += [getattr(NU_W0WA, p) for p in BACKEND_PARAMETERS]
    np.testing.assert_array_equal(leaves, expected)


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_jit_vmap_over_cosmologies_matches_loop(nonlinear: bool) -> None:
    cosmos = [replace(c, nonlinear=nonlinear) for c in [LCDM, NU_W0WA, HEAVY_NU]]
    batch = jax.tree.map(lambda *xs: jnp.array(xs), *cosmos)
    k = jnp.geomspace(0.01, 3.0, 20)
    fn = jax.jit(jax.vmap(lambda c: c.matter_power_spectrum(k, 0.7)))
    expected = [c.matter_power_spectrum(k, 0.7) for c in cosmos]
    np.testing.assert_allclose(fn(batch), np.array(expected), rtol=1e-10)


def _shifted(cosmo: BaccoEmu, name: str, step: float) -> BaccoEmu:
    if name in BACKEND_PARAMETERS:
        return replace(cosmo, **{name: getattr(cosmo, name) + step})
    bg = cosmo.background
    return replace(cosmo, background=replace(bg, **{name: getattr(bg, name) + step}))


@pytest.mark.parametrize("name", COSMOLOGY_PARAMETERS + BACKEND_PARAMETERS)
def test_gradient_matches_finite_differences(name: str) -> None:
    k = jnp.geomspace(0.02, 3.0, 30)
    z = jnp.array([0.3, 1.1])[:, None]

    cosmo = replace(NU_W0WA, nonlinear=True)

    def log_pk(c: BaccoEmu) -> jax.Array:
        return jnp.log(c.matter_power_spectrum(k, z))

    jac = jax.jacfwd(log_pk)(cosmo)
    grad = getattr(jac if name in BACKEND_PARAMETERS else jac.background, name)
    step = 1e-6
    fd = (
        log_pk(_shifted(cosmo, name, step)) - log_pk(_shifted(cosmo, name, -step))
    ) / (2 * step)
    np.testing.assert_allclose(grad, fd, rtol=1e-3, atol=1e-3 * np.max(np.abs(fd)))


def test_check_bounds_inside_range_passes() -> None:
    replace(LCDM, nonlinear=True).check_bounds(K_H * little_h(LCDM), Z)
    LCDM.check_bounds(z=2.0)


@pytest.mark.parametrize(
    ("cosmo", "k", "z", "match"),
    [
        (
            backend(BACCO["lcdm"] | {"omega_cold": 0.5}, nonlinear=True),
            None,
            0.0,
            "omega_cold",
        ),
        (replace(LCDM, nonlinear=True), None, 2.0, "expfactor"),
        (replace(LCDM, nonlinear=True), 10.0, 0.0, "k/h"),
        (LCDM, 1e-6, 0.0, "k/h"),
    ],
    ids=["parameter", "redshift", "k_max", "k_min"],
)
def test_check_bounds_outside_range_raises(
    cosmo: BaccoEmu, k: float | None, z: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        cosmo.check_bounds(k, z)


def test_no_amplitude_raises() -> None:
    with pytest.raises(ValueError, match="ln10as"):
        replace(LCDM, sigma8_cold=None).matter_power_spectrum(0.1, 0.0)


def test_sigma8_cold_overrides_ln10as() -> None:
    with_ln10as = replace(LCDM, background=replace(LCDM.background, ln10as=3.3))
    np.testing.assert_allclose(
        with_ln10as.matter_power_spectrum(0.1, 0.0),
        LCDM.matter_power_spectrum(0.1, 0.0),
    )
