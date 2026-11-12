"""Conformance checks for implementations of the protocols.

Third-party backends can call these from their own test suites::

    from cosmograd.testing import check_background

    def test_my_background():
        check_background(MyBackground(...), z=jnp.linspace(0.0, 3.0, 50))

The checks assume concrete (non-traced) inputs and should run in double
precision; enable ``jax_enable_x64`` in your test configuration.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Real

from .background import Background
from .perturbations import Perturbations

__all__ = ["check_background", "check_perturbations"]

_BACKGROUND_METHODS: tuple[str, ...] = (
    "hubble_parameter",
    "matter_density_parameter",
    "comoving_distance",
    "transverse_comoving_distance",
    "luminosity_distance",
    "angular_diameter_distance",
    "differential_comoving_volume",
)


def _check_leaves(cosmo: object) -> None:
    """Check that ``cosmo`` has pytree leaves and that all are floating point."""
    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves, (
        "the cosmology has no pytree leaves, so nothing can be differentiated"
    )
    for leaf in leaves:
        assert jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact), (
            f"pytree leaf {leaf!r} is not floating point; static settings "
            "belong in meta fields"
        )


def check_background(
    cosmo: Background,
    z: Real[ArrayLike, "*shape"],
    *,
    rtol: float = 1e-6,
) -> None:
    """Check that ``cosmo`` honours the :class:`~cosmograd.Background` contract.

    Checks that ``cosmo``

    - is an instance of :class:`~cosmograd.Background` and a pytree with at
      least one leaf, all of them floating point;
    - returns finite arrays with the shape of ``z`` from every method, and a
      finite scalar ``hubble_constant``;
    - satisfies :math:`d_L = (1 + z)^2 d_A`;
    - inverts :meth:`comoving_distance` with
      :meth:`redshift_at_comoving_distance`;
    - gives the same results under :func:`jax.jit`;
    - has a finite :func:`jax.jacfwd` with respect to itself, shaped like
      ``cosmo``.

    Parameters
    ----------
    cosmo : Background
        The implementation to check.
    z : ArrayLike
        Redshifts to evaluate at, non-negative and inside the range ``cosmo``
        documents. Use at least one non-zero value.
    rtol : float, default 1e-6
        Relative tolerance of the numerical identities.

    Raises
    ------
    AssertionError
        If any check fails.
    """
    z = jnp.asarray(z)

    assert isinstance(cosmo, Background), f"{type(cosmo).__name__} is not a Background"

    _check_leaves(cosmo)

    h0 = jnp.asarray(cosmo.hubble_constant)
    assert h0.shape == (), f"hubble_constant has shape {h0.shape}, expected ()"
    assert jnp.isfinite(h0), "hubble_constant is not finite"

    for name in _BACKGROUND_METHODS:
        out = getattr(cosmo, name)(z)
        assert out.shape == z.shape, f"{name} has shape {out.shape}, expected {z.shape}"
        assert jnp.all(jnp.isfinite(out)), f"{name} returned non-finite values"

    np.testing.assert_allclose(
        cosmo.luminosity_distance(z),
        (1 + z) ** 2 * cosmo.angular_diameter_distance(z),
        rtol=rtol,
        err_msg="luminosity and angular diameter distances are inconsistent",
    )

    chi = cosmo.comoving_distance(z)
    np.testing.assert_allclose(
        cosmo.redshift_at_comoving_distance(chi),
        z,
        rtol=rtol,
        atol=rtol,
        err_msg="redshift_at_comoving_distance does not invert comoving_distance",
    )

    np.testing.assert_allclose(
        jax.jit(lambda c: c.comoving_distance(z))(cosmo),
        chi,
        rtol=rtol,
        err_msg="comoving_distance differs under jit",
    )

    jac = jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)
    assert jax.tree_util.tree_structure(jac) == jax.tree_util.tree_structure(cosmo), (
        "the Jacobian is not shaped like the cosmology"
    )
    for leaf in jax.tree_util.tree_leaves(jac):
        assert jnp.all(jnp.isfinite(leaf)), "the Jacobian has non-finite entries"


def check_perturbations(
    cosmo: Perturbations,
    k: Real[ArrayLike, "*#shape"],
    z: Real[ArrayLike, "*#shape"],
    *,
    rtol: float = 1e-6,
    growth_rate_rtol: float = 0.1,
    linear_cosmo: Perturbations | None = None,
) -> None:
    """Check that ``cosmo`` honours the :class:`~cosmograd.Perturbations` contract.

    Checks that ``cosmo``

    - is an instance of :class:`~cosmograd.Perturbations` and a pytree with at
      least one leaf, all of them floating point, and that its
      :attr:`~cosmograd.Perturbations.background` is a
      :class:`~cosmograd.Background`;
    - returns finite arrays with the broadcast shape of ``k`` and ``z`` from
      every method, positive for the power spectra, and the shape of ``z``
      from the growth methods without ``k``;
    - broadcasts ``k`` and ``z`` elementwise;
    - normalises the growth function to 1 at :math:`z = 0`;
    - when ``linear_cosmo`` is supplied, satisfies
      :math:`D^2(z, k) = P_\\mathrm{lin}(k, z) / P_\\mathrm{lin}(k, 0)`;
    - returns a growth rate equal to :math:`\\mathrm{d}\\ln D / \\mathrm{d}\\ln a`,
      from a one-sided finite difference towards higher redshift;
    - gives the same results under :func:`jax.jit`;
    - has a finite :func:`jax.jacfwd` with respect to itself, shaped like
      ``cosmo``, for its configured matter spectrum.

    Parameters
    ----------
    cosmo : Perturbations
        The implementation to check.
    k : ArrayLike
        Wavenumbers in 1/Mpc, inside the range ``cosmo`` documents.
    z : ArrayLike
        Redshifts, broadcastable against ``k``, non-negative and inside the
        range ``cosmo`` documents for the spectra being checked. Use at least
        one non-zero value. The growth-rate check also evaluates the growth
        function up to :math:`\\ln(1 + z)` + 0.1.
    rtol : float, default 1e-6
        Relative tolerance of the exact identities.
    growth_rate_rtol : float, default 0.1
        Relative tolerance of the growth-rate identity. It checks the
        convention (derivative in :math:`\\ln a`, sign), not accuracy:
        emulated growth functions are not smooth enough for a tight
        finite-difference comparison.
    linear_cosmo : Perturbations, optional
        Matching instance configured for linear power, used to check the
        growth-to-power identity. May be ``cosmo`` itself for linear instances.
        When omitted, this identity is not checked.

    Raises
    ------
    AssertionError
        If any check fails.
    """
    k, z = jnp.asarray(k), jnp.asarray(z)
    shape = jnp.broadcast_shapes(k.shape, z.shape)

    assert isinstance(cosmo, Perturbations), (
        f"{type(cosmo).__name__} is not a Perturbations"
    )
    assert isinstance(cosmo.background, Background), "background is not a Background"
    _check_leaves(cosmo)

    pk = cosmo.matter_power_spectrum(k, z)
    name = "matter_power_spectrum"
    assert pk.shape == shape, f"{name} has shape {pk.shape}, expected {shape}"
    assert jnp.all(jnp.isfinite(pk)), f"{name} returned non-finite values"
    assert jnp.all(pk > 0), f"{name} returned non-positive values"

    k_b, z_b = jnp.broadcast_arrays(k, z)
    np.testing.assert_allclose(
        cosmo.matter_power_spectrum(k_b.ravel(), z_b.ravel()),
        pk.ravel(),
        rtol=rtol,
        err_msg=f"{name} does not broadcast k and z elementwise",
    )

    for name in ("growth_function", "growth_rate"):
        method = getattr(cosmo, name)
        for k_arg, expected in ((k, shape), (None, z.shape)):
            out = method(z, k=k_arg)
            assert out.shape == expected, (
                f"{name}(k={'None' if k_arg is None else 'k'}) has shape "
                f"{out.shape}, expected {expected}"
            )
            assert jnp.all(jnp.isfinite(out)), f"{name} returned non-finite values"

    zero = jnp.zeros_like(z)
    np.testing.assert_allclose(
        cosmo.growth_function(zero, k=k),
        jnp.ones(shape),
        rtol=rtol,
        err_msg="growth_function is not 1 at z = 0",
    )

    if linear_cosmo is not None:
        growth = cosmo.growth_function(z, k=k)
        np.testing.assert_allclose(
            growth**2,
            linear_cosmo.matter_power_spectrum(k, z)
            / linear_cosmo.matter_power_spectrum(k, zero),
            rtol=rtol,
            err_msg="growth_function**2 is not the ratio of linear power spectra",
        )

    # Second-order one-sided difference in ln a, stepping towards higher z so
    # that no redshift is negative.
    step = 0.05
    ln_d = [
        jnp.log(cosmo.growth_function(jnp.expm1(jnp.log1p(z) + i * step), k=k))
        for i in range(3)
    ]
    np.testing.assert_allclose(
        cosmo.growth_rate(z, k=k),
        (3 * ln_d[0] - 4 * ln_d[1] + ln_d[2]) / (2 * step),
        rtol=growth_rate_rtol,
        err_msg="growth_rate is not d ln D / d ln a",
    )

    primordial = cosmo.primordial_power_spectrum(k)
    assert primordial.shape == k.shape, (
        f"primordial_power_spectrum has shape {primordial.shape}, expected {k.shape}"
    )
    assert jnp.all(jnp.isfinite(primordial) & (primordial > 0)), (
        "primordial_power_spectrum returned non-finite or non-positive values"
    )

    np.testing.assert_allclose(
        jax.jit(lambda c: c.matter_power_spectrum(k, z))(cosmo),
        cosmo.matter_power_spectrum(k, z),
        rtol=rtol,
        err_msg="matter_power_spectrum differs under jit",
    )

    jac = jax.jacfwd(lambda c: c.matter_power_spectrum(k, z))(cosmo)
    assert jax.tree_util.tree_structure(jac) == jax.tree_util.tree_structure(cosmo), (
        "the Jacobian is not shaped like the cosmology"
    )
    for leaf in jax.tree_util.tree_leaves(jac):
        assert jnp.all(jnp.isfinite(leaf)), "the Jacobian has non-finite entries"
