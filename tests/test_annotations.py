"""Shape and dtype contracts under the default pytest import hook."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real, TypeCheckError, jaxtyped

from cosmograd import FlatLCDM

from ._typecheck import typecheck

BACKGROUND_METHODS = (
    "efunc",
    "hubble_parameter",
    "matter_density_parameter",
    "comoving_distance",
    "redshift_at_comoving_distance",
    "transverse_comoving_distance",
    "luminosity_distance",
    "angular_diameter_distance",
    "differential_comoving_volume",
)


@pytest.mark.parametrize("method", BACKGROUND_METHODS)
@pytest.mark.parametrize(
    "value",
    [
        1,
        1.0,
        np.array(1),
        np.array(1.0),
        jnp.array(1),
        jnp.array(1.0),
        np.array([0, 1, 2]),
        np.array([0.0, 1.0, 2.0]),
        jnp.array([0, 1, 2]),
        jnp.array([0.0, 1.0, 2.0]),
        jnp.ones((2, 3), dtype=int),
        jnp.ones((2, 3)),
    ],
)
def test_background_input_shapes(method: str, value: ArrayLike) -> None:
    cosmo = FlatLCDM(h=0.67, omega_m=0.31)
    result = getattr(cosmo, method)(value)
    assert isinstance(result, Array)
    assert result.shape == np.shape(value)
    assert jnp.issubdtype(result.dtype, jnp.floating)


@pytest.mark.parametrize("name", ["hubble_constant", "omega_r", "omega_de"])
def test_scalar_properties(name: str) -> None:
    result = getattr(FlatLCDM(h=0.67, omega_m=0.31), name)
    assert isinstance(result, Array)
    assert result.shape == ()
    assert jnp.issubdtype(result.dtype, jnp.floating)


@pytest.mark.parametrize("method", BACKGROUND_METHODS)
@pytest.mark.parametrize("value", [1 + 1j, np.array([1j]), jnp.array([1j])])
def test_import_hook_rejects_complex(method: str, value: ArrayLike) -> None:
    cosmo = FlatLCDM(h=0.67, omega_m=0.31)
    with pytest.raises(TypeCheckError):
        getattr(cosmo, method)(value)


def test_checked_return_shape() -> None:
    @jaxtyped(typechecker=typecheck)
    def wrong_shape(z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:  # noqa: F722
        return jnp.zeros((2, 3))

    with pytest.raises(TypeCheckError):
        wrong_shape(jnp.ones(3))


@jaxtyped(typechecker=typecheck)
def broadcast_signature(
    k: Real[ArrayLike, "*#shape"],  # noqa: F722
    z: Real[ArrayLike, "*#shape"],  # noqa: F722
) -> Float[Array, "*shape"]:  # noqa: F722
    """Exercise the perturbations contract without implementing a backend."""
    return jnp.asarray(k, dtype=float) + jnp.asarray(z)


@pytest.mark.parametrize(
    ("k", "z", "shape"),
    [
        (1, jnp.ones(3), (3,)),
        (jnp.ones(3), 1.0, (3,)),
        (jnp.ones((2, 1)), jnp.ones((1, 3)), (2, 3)),
    ],
)
def test_broadcast_shapes(k: ArrayLike, z: ArrayLike, shape: tuple[int, ...]) -> None:
    result = broadcast_signature(k, z)
    assert result.shape == shape
    assert jnp.issubdtype(result.dtype, jnp.floating)
