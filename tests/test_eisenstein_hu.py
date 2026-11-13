"""EH98 physical identities and native JAX transformation behaviour."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from cosmograd import EisensteinHu, FlatLCDM, FlatW0WaCDM, Perturbations
from cosmograd.perturbations import EisensteinHu as ExportedEisensteinHu

from .conftest import CosmologyCase, camb_results, cosmology_params


@pytest.fixture(
    params=[
        EisensteinHu(FlatLCDM(H0=67.36, omega_m=0.3153), 0.0493, 0.9649, 3.044),
        EisensteinHu(FlatLCDM(H0=72.0, omega_m=0.25), 0.04, 1.0, 3.1),
        EisensteinHu(FlatLCDM(H0=62.0, omega_m=0.4, t_cmb=2.8), 0.07, 0.94, 2.9),
    ],
    ids=["planck", "low-matter", "high-matter"],
)
def eh(request: pytest.FixtureRequest) -> EisensteinHu:
    return request.param


def test_background_and_pytree(eh: EisensteinHu) -> None:
    bg = eh.background
    cosmo = EisensteinHu(bg, eh.omega_b, eh.n_s, eh.ln10as)
    assert cosmo.background is bg
    assert isinstance(cosmo, Perturbations)
    assert ExportedEisensteinHu is EisensteinHu
    np.testing.assert_array_equal(
        jax.tree_util.tree_leaves(cosmo),
        [bg.H0, bg.omega_m, eh.omega_b, eh.n_s, eh.ln10as],
    )
    with pytest.raises(FrozenInstanceError):
        setattr(cosmo, "n_s", 1.0)


def test_primordial_pivot(eh: EisensteinHu) -> None:
    np.testing.assert_allclose(
        eh.primordial_power_spectrum(0.05), 1e-10 * np.exp(eh.ln10as)
    )


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_amplitude_scaling(eh: EisensteinHu, factor: float) -> None:
    scaled = replace(eh, ln10as=eh.ln10as + np.log(factor))
    k, z = jnp.array([0.001, 0.05, 0.2]), jnp.array([0.0, 1.0, 10.0])
    np.testing.assert_allclose(
        scaled.matter_power_spectrum(k, z), factor * eh.matter_power_spectrum(k, z)
    )


def test_growth_normalisation_and_scale_independence(eh: EisensteinHu) -> None:
    k = jnp.array([0.2, 0.001, 0.05])
    z = jnp.array([10.0, 0.0, 1.0])
    np.testing.assert_allclose(eh.growth_function(0.0, k=k), np.ones(3))
    np.testing.assert_allclose(
        eh.growth_function(z[:, None], k=k),
        np.broadcast_to(eh.growth_function(z)[:, None], (3, 3)),
    )
    np.testing.assert_allclose(
        eh.growth_rate(z[:, None], k=k),
        np.broadcast_to(eh.growth_rate(z)[:, None], (3, 3)),
    )
    np.testing.assert_allclose(
        eh.matter_power_spectrum(k, z) / eh.matter_power_spectrum(k, 0.0),
        eh.growth_function(z) ** 2,
    )


@pytest.mark.parametrize("z", [0.0, 0.5, 3.0, 10.0])
def test_growth_rate_derivative(eh: EisensteinHu, z: float) -> None:
    derivative = jax.grad(lambda x: jnp.log(eh.growth_function(x)))(z)
    np.testing.assert_allclose(eh.growth_rate(z), -(1 + z) * derivative, rtol=1e-6)


@pytest.mark.parametrize("z", [0.0, 0.5, 3.0, 10.0])
def test_matter_only_growth(z: float) -> None:
    eh = EisensteinHu(FlatLCDM(H0=70.0, omega_m=1.0), 0.05, 0.96, 3.0)
    np.testing.assert_allclose(eh.growth_function(z), 1 / (1 + z), rtol=1e-10)
    np.testing.assert_allclose(eh.growth_rate(z), 1.0, rtol=1e-10)


@pytest.mark.parametrize(
    "method",
    [
        "growth_function",
        "growth_rate",
        "primordial_power_spectrum",
        "matter_power_spectrum",
    ],
)
def test_jit_and_vmap(eh: EisensteinHu, method: str) -> None:
    # Unsorted inputs also exercise elementwise batching over evaluation points.
    x = jnp.array([0.2, 0.01, 0.07])

    def evaluate(c: EisensteinHu, value: Array) -> Array:
        if method == "matter_power_spectrum":
            return c.matter_power_spectrum(value, 1.0)
        return getattr(c, method)(value)

    expected = evaluate(eh, x)
    np.testing.assert_allclose(jax.jit(evaluate)(eh, x), expected)
    np.testing.assert_allclose(jax.vmap(evaluate, in_axes=(None, 0))(eh, x), expected)
    second = replace(eh, background=replace(eh.background, omega_m=0.33), n_s=0.97)
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), eh, second)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate, in_axes=(0, None)))(batch, x),
        jnp.stack([expected, evaluate(second, x)]),
    )


def test_parameter_gradients(eh: EisensteinHu) -> None:
    gradient = jax.jit(jax.grad(lambda c: jnp.log(c.matter_power_spectrum(0.1, 0.5))))(
        eh
    )
    assert isinstance(gradient.background, FlatLCDM)
    leaves = jnp.array(jax.tree_util.tree_leaves(gradient))
    assert leaves.shape == (5,)
    assert jnp.all(jnp.isfinite(leaves) & (leaves != 0))
    np.testing.assert_allclose(gradient.ln10as, 1.0)


def test_nonlinear_rejected(eh: EisensteinHu) -> None:
    with pytest.raises(NotImplementedError, match="linear power only"):
        replace(eh, nonlinear=True)


def test_default_w0wa_background_matches_lcdm() -> None:
    k, z = jnp.geomspace(1e-3, 1.0, 5)[:, None], jnp.array([0.0, 1.0, 3.0])
    lcdm = EisensteinHu(FlatLCDM(H0=67.36, omega_m=0.3153), 0.0493, 0.9649, 3.044)
    w0wa = replace(lcdm, background=FlatW0WaCDM(H0=67.36, omega_m=0.3153))
    np.testing.assert_allclose(
        w0wa.matter_power_spectrum(k, z), lcdm.matter_power_spectrum(k, z)
    )


def test_dark_energy_changes_growth_only() -> None:
    k = jnp.geomspace(1e-3, 1.0, 5)
    lcdm = EisensteinHu(FlatLCDM(H0=67.36, omega_m=0.3153), 0.0493, 0.9649, 3.044)
    background = FlatW0WaCDM(H0=67.36, omega_m=0.3153, w0=-0.8, wa=0.2)
    w0wa = replace(lcdm, background=background)
    ratio = w0wa.matter_power_spectrum(k, 0.5) / lcdm.matter_power_spectrum(k, 0.5)
    # Earlier dark energy suppresses growth, by the same factor on every scale.
    assert ratio[0] < 1
    np.testing.assert_allclose(ratio, ratio[0])


def _sigma8(eh: EisensteinHu) -> Array:
    """RMS linear fluctuation in 8 Mpc/h spheres, by top-hat quadrature in ln k."""
    h = jnp.asarray(eh.background.H0) / 100
    ln_k = jnp.linspace(jnp.log(1e-5), jnp.log(50.0), 4000)
    k = jnp.exp(ln_k)
    x = k * 8 / h
    window = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
    integrand = k**3 * eh.linear_matter_power_spectrum(k, 0.0) * window**2
    return jnp.sqrt(jnp.trapezoid(integrand, ln_k) / (2 * jnp.pi**2))


def _jax_cosmo_power(case: CosmologyCase, sigma8: Array, k: Array, z: float) -> Array:
    """jax-cosmo's EH98 spectrum in 1/Mpc and Mpc^3, at the given sigma8."""
    jc = pytest.importorskip("jax_cosmo")
    transfer = pytest.importorskip("jax_cosmo.transfer")
    bg = case.background
    h = jnp.asarray(bg.H0) / 100
    cosmo = jc.Cosmology(
        Omega_c=bg.omega_m - case.omega_b,
        Omega_b=case.omega_b,
        h=h,
        n_s=case.n_s,
        sigma8=sigma8,
        Omega_k=0.0,
        w0=bg.w0,
        wa=bg.wa,
    )
    # jax-cosmo works in h/Mpc and (Mpc/h)^3.
    pk = jc.power.linear_matter_power(
        cosmo,
        k / h,
        a=1 / (1 + z),
        transfer_fn=transfer.Eisenstein_Hu,
        type="eisenhu_osc",
    )
    return pk / h**3


def _perturbations(case: CosmologyCase) -> EisensteinHu:
    assert isinstance(case.background, FlatLCDM | FlatW0WaCDM)
    return EisensteinHu(case.background, case.omega_b, case.n_s, case.ln10as)


@pytest.mark.slow
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("case", cosmology_params("flat"))
def test_against_jax_cosmo(case: CosmologyCase, z: float) -> None:
    eh = _perturbations(case)
    k = jnp.geomspace(1e-3, 1.0, 12)
    # jax-cosmo normalises through sigma8, so hand it ours.
    expected = _jax_cosmo_power(case, _sigma8(eh), k, z)
    np.testing.assert_allclose(
        eh.linear_matter_power_spectrum(k, z), expected, rtol=1e-2
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["omega_m", "omega_b"])
@pytest.mark.parametrize("case", cosmology_params("flat_lcdm"))
def test_shape_gradient_against_jax_cosmo(case: CosmologyCase, name: str) -> None:
    # The ratio P(k) / P(k_ref) is independent of the amplitude, so each code
    # can be differentiated without coupling sigma8 to the parameters.
    k = jnp.array([0.02, 0.1, 0.5])
    sigma8 = _sigma8(_perturbations(case))

    def ours(omega_m: Array, omega_b: Array) -> Array:
        bg = replace(_perturbations(case).background, omega_m=omega_m)
        eh = EisensteinHu(bg, omega_b, case.n_s, case.ln10as)
        return jnp.log(eh.linear_matter_power_spectrum(k, 0.0))

    def theirs(omega_m: Array, omega_b: Array) -> Array:
        replaced = replace(
            case, omega_b=omega_b, background=replace(case.background, omega_m=omega_m)
        )
        return jnp.log(_jax_cosmo_power(replaced, sigma8, k, 0.0))

    argnum = ("omega_m", "omega_b").index(name)
    point = (jnp.asarray(case.background.omega_m), jnp.asarray(case.omega_b))
    ours_jac = jax.jacfwd(ours, argnums=argnum)(*point)
    theirs_jac = jax.jacfwd(theirs, argnums=argnum)(*point)
    # Remove the amplitude: differences of log P between wavenumbers.
    np.testing.assert_allclose(
        jnp.diff(ours_jac), jnp.diff(theirs_jac), rtol=5e-2, atol=1e-2
    )


@pytest.mark.slow
@pytest.mark.parametrize("z", [0.0, 1.0])
@pytest.mark.parametrize("case", cosmology_params("flat"))
def test_against_camb(case: CosmologyCase, z: float) -> None:
    results = camb_results(case, power=True, kmax=2.0, zs=(0.0, 1.0))
    interpolator = results.get_matter_power_interpolator(
        nonlinear=False, hubble_units=False, k_hunit=False
    )
    k = np.geomspace(1e-3, 1.0, 40)
    # EH98 is a fitting formula and cannot match a Boltzmann code more closely
    # than a few per cent (the BAO phase shifts it most); it is a sanity bound on
    # shape and amplitude.
    np.testing.assert_allclose(
        _perturbations(case).linear_matter_power_spectrum(k, z),
        interpolator.P(z, k),
        rtol=8e-2,
    )
