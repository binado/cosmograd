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


def check_background(
    cosmo: Background,
    z: Real[ArrayLike, "*shape"],  # noqa: F722
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

    leaves = jax.tree_util.tree_leaves(cosmo)
    assert leaves, (
        "the cosmology has no pytree leaves, so nothing can be differentiated"
    )
    for leaf in leaves:
        assert jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact), (
            f"pytree leaf {leaf!r} is not floating point; static settings "
            "belong in meta fields"
        )

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
    k: Real[ArrayLike, "*#shape"],  # noqa: F722
    z: Real[ArrayLike, "*#shape"],  # noqa: F722
) -> None:
    """Check that ``cosmo`` honours the :class:`~cosmograd.Perturbations` contract.

    Not implemented yet; it will land with the first native perturbations
    implementation (Eisenstein-Hu).

    Parameters
    ----------
    cosmo : Perturbations
        The implementation to check.
    k : ArrayLike
        Wavenumbers in 1/Mpc.
    z : ArrayLike
        Redshifts, broadcastable against ``k``.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError
