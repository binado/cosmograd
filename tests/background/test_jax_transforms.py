"""Background methods behave under ``jit`` and ``vmap`` as they do eagerly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmograd import cosmologies as standard

from ..cosmologies import Cosmology, cosmology_params

Z = jnp.linspace(0.0, 5.0, 25)
METHODS = [
    "hubble_parameter",
    "matter_density_parameter",
    "comoving_distance",
    "transverse_comoving_distance",
    "luminosity_distance",
    "angular_diameter_distance",
    "differential_comoving_volume",
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cosmo", cosmology_params())
def test_method_under_jit_matches_eager(cosmo: Cosmology, method: str) -> None:
    np.testing.assert_allclose(
        jax.jit(lambda c: getattr(c, method)(Z))(cosmo), getattr(cosmo, method)(Z)
    )


@pytest.fixture
def batch() -> tuple[list, object]:
    """Three cosmologies of one class, and the same stacked into one pytree."""
    cosmos = [standard.Planck13, standard.Planck15, standard.Planck18]
    return cosmos, jax.tree.map(lambda *xs: jnp.stack(xs), *cosmos)


@pytest.mark.parametrize("method", ["comoving_distance", "luminosity_distance"])
def test_method_under_vmap_over_cosmologies_matches_loop(
    batch: tuple[list, object], method: str
) -> None:
    cosmos, stacked = batch
    actual = jax.jit(jax.vmap(lambda c: getattr(c, method)(Z)))(stacked)
    expected = [getattr(c, method)(Z) for c in cosmos]
    np.testing.assert_allclose(actual, np.array(expected))


def test_inverse_distance_under_vmap_over_cosmologies_matches_loop(
    batch: tuple[list, object],
) -> None:
    cosmos, stacked = batch
    chi = jnp.stack([c.comoving_distance(Z) for c in cosmos])
    actual = jax.jit(jax.vmap(lambda c, d: c.redshift_at_comoving_distance(d)))(
        stacked, chi
    )
    np.testing.assert_allclose(actual, jnp.broadcast_to(Z, actual.shape), atol=1e-6)
