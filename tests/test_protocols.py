from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from cosmograd import Background, FlatLCDM, Perturbations


@dataclass(frozen=True)
class Toy(Background):
    """Static universe with constant H, so chi = z / H."""

    h0: float = 1e-3

    @property
    def hubble_constant(self) -> Array:
        return jnp.asarray(self.h0)

    def hubble_parameter(self, z: ArrayLike) -> Array:
        return jnp.full_like(jnp.asarray(z, dtype=float), self.h0)

    def matter_density_parameter(self, z: ArrayLike) -> Array:
        return jnp.ones_like(jnp.asarray(z, dtype=float))

    def comoving_distance(self, z: ArrayLike) -> Array:
        return jnp.asarray(z) / self.h0

    def redshift_at_comoving_distance(self, chi: ArrayLike) -> Array:
        return jnp.asarray(chi) * self.h0


def test_provided_methods() -> None:
    toy = Toy()
    z = jnp.array([0.5, 1.0, 2.0])
    chi = z / toy.h0
    np.testing.assert_allclose(toy.transverse_comoving_distance(z), chi)
    np.testing.assert_allclose(toy.luminosity_distance(z), (1 + z) * chi)
    np.testing.assert_allclose(toy.angular_diameter_distance(z), chi / (1 + z))
    np.testing.assert_allclose(toy.differential_comoving_volume(z), chi**2 / toy.h0)


def test_isinstance() -> None:
    cosmo = FlatLCDM(H0=67.0, omega_m=0.31)
    assert isinstance(cosmo, Background)
    assert not isinstance(cosmo, Perturbations)
