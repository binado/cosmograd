"""Independent implementations of the background protocol, to compare against.

None of these classes inherits from :class:`~cosmograd.Background`, so no
method comes from cosmograd's own defaults. Each answers with the method names
of the protocol and can stand in for a cosmograd background in a comparison.

- :class:`EinsteinDeSitter`, :class:`Milne` and :class:`DeSitter` are exact
  closed forms. They are pytrees in ``H0``, so their gradients are exact too.
- :class:`CAMBBackground` answers with CAMB's own functions.
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from cosmograd.constants import C_KM_S

from ..cosmologies import Cosmology


@dataclass(frozen=True)
class _ClosedForm:
    """A matter, curvature and cosmological-constant free universe in closed form.

    Radiation is absent. The derived distances are written as their textbook
    definitions in terms of the transverse comoving distance and the Hubble rate.
    """

    H0: Real[ArrayLike, ""]
    """Hubble constant in km/s/Mpc."""

    @property
    def hubble_constant(self) -> Float[Array, ""]:
        return jnp.asarray(self.H0) / C_KM_S

    def luminosity_distance(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return (1 + jnp.asarray(z)) * self.transverse_comoving_distance(z)

    def angular_diameter_distance(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return self.transverse_comoving_distance(z) / (1 + jnp.asarray(z))

    def differential_comoving_volume(
        self, z: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return self.transverse_comoving_distance(z) ** 2 / self.hubble_parameter(z)

    def transverse_comoving_distance(
        self, z: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        raise NotImplementedError

    def hubble_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        raise NotImplementedError


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EinsteinDeSitter(_ClosedForm):
    r"""Flat, matter only: :math:`H = H_0 (1 + z)^{3/2}`."""

    def hubble_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return self.hubble_constant * (1 + jnp.asarray(z, dtype=float)) ** 1.5

    def matter_density_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return jnp.ones_like(jnp.asarray(z, dtype=float))

    def comoving_distance(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return 2 / self.hubble_constant * (1 - (1 + jnp.asarray(z)) ** -0.5)

    def redshift_at_comoving_distance(
        self, chi: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return (1 - self.hubble_constant * jnp.asarray(chi) / 2) ** -2 - 1

    def transverse_comoving_distance(
        self, z: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return self.comoving_distance(z)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Milne(_ClosedForm):
    r"""Empty and open: :math:`H = H_0 (1 + z)`, :math:`\Omega_k = 1`."""

    def hubble_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return self.hubble_constant * (1 + jnp.asarray(z, dtype=float))

    def matter_density_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return jnp.zeros_like(jnp.asarray(z, dtype=float))

    def comoving_distance(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return jnp.log1p(jnp.asarray(z)) / self.hubble_constant

    def redshift_at_comoving_distance(
        self, chi: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return jnp.expm1(self.hubble_constant * jnp.asarray(chi))

    def transverse_comoving_distance(
        self, z: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return jnp.sinh(jnp.log1p(jnp.asarray(z))) / self.hubble_constant


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DeSitter(_ClosedForm):
    r"""Flat, cosmological constant only: :math:`H = H_0`."""

    def hubble_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return jnp.full_like(jnp.asarray(z, dtype=float), self.hubble_constant)

    def matter_density_parameter(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return jnp.zeros_like(jnp.asarray(z, dtype=float))

    def comoving_distance(self, z: Real[ArrayLike, "*s"]) -> Float[Array, "*s"]:
        return jnp.asarray(z) / self.hubble_constant

    def redshift_at_comoving_distance(
        self, chi: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return self.hubble_constant * jnp.asarray(chi)

    def transverse_comoving_distance(
        self, z: Real[ArrayLike, "*s"]
    ) -> Float[Array, "*s"]:
        return self.comoving_distance(z)


@cache
def _camb_results(
    key: tuple[float, ...], power: bool, kmax: float, zs: tuple[float, ...]
) -> Any:
    camb = pytest.importorskip("camb")
    h0, omega_m, omega_k, w0, wa, n_eff, t_cmb, omega_b, n_s, ln10as = key
    h2 = (h0 / 100) ** 2
    # cosmograd treats massive neutrinos as matter, so match with mnu = 0 and
    # put the whole of omega_m into baryons plus CDM.
    pars = camb.set_params(
        H0=h0,
        ombh2=omega_b * h2,
        omch2=(omega_m - omega_b) * h2,
        omk=omega_k,
        mnu=0.0,
        nnu=n_eff,
        TCMB=t_cmb,
        ns=n_s,
        As=math.exp(ln10as) * 1e-10,
        kmax=kmax,
    )
    if (w0, wa) != (-1.0, 0.0):
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
    if not power:
        return camb.get_background(pars)
    pars.set_matter_power(redshifts=list(zs), kmax=kmax)
    return camb.get_results(pars)


def camb_results(
    cosmo: Cosmology,
    *,
    power: bool = False,
    kmax: float = 2.0,
    zs: tuple[float, ...] = (0.0,),
) -> Any:
    """CAMB results for a cosmology, cached because the runs are the slow part.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology to compute.
    power : bool, default False
        Also compute the linear matter power spectrum. Without it only the
        background is solved.
    kmax : float, default 2.0
        Largest wavenumber in 1/Mpc, when ``power`` is set.
    zs : tuple of float, default (0.0,)
        Redshifts at which power is stored, when ``power`` is set.

    Returns
    -------
    camb.results.CAMBdata
        Skips the test when CAMB is not installed.

    Notes
    -----
    ``omega_m`` in cosmograd folds massive neutrinos into matter, so CAMB is
    run with ``mnu = 0`` and ``omch2 = (omega_m - omega_b) h^2``.
    """
    key = tuple(
        float(np.asarray(getattr(cosmo, name)))
        for name in (
            "H0",
            "omega_m",
            "omega_k",
            "w0",
            "wa",
            "n_eff",
            "t_cmb",
            "omega_b",
            "n_s",
            "ln10as",
        )
    )
    return _camb_results(key, power, kmax if power else 0.0, zs if power else ())


@dataclass(frozen=True)
class CAMBBackground:
    """The background protocol, answered by CAMB.

    Every method calls CAMB. Where CAMB has no function of the same name the
    quantity is built from CAMB's own Hubble rate and distances.
    """

    results: Any
    """The ``camb.results.CAMBdata`` of the matched cosmology."""

    @classmethod
    def from_background(cls, cosmo: Cosmology) -> "CAMBBackground":
        """Run CAMB for the cosmology of a cosmograd background.

        Parameters
        ----------
        cosmo : Cosmology
            Any background holding every parameter, as the in-tree ones do.

        Returns
        -------
        CAMBBackground
            Skips the test when CAMB is not installed.
        """
        return cls(camb_results(cosmo))

    @property
    def hubble_constant(self) -> Any:
        return self.results.h_of_z(0.0)

    @property
    def omega_r(self) -> Any:
        """Photons plus massless neutrinos today."""
        return self.results.get_Omega("photon") + self.results.get_Omega("neutrino")

    def hubble_parameter(self, z: Any) -> Any:
        return self.results.h_of_z(z)

    def matter_density_parameter(self, z: Any) -> Any:
        # mnu = 0, so there is no massive neutrino component to add.
        return self.results.get_Omega("cdm", z=z) + self.results.get_Omega(
            "baryon", z=z
        )

    def comoving_distance(self, z: Any) -> Any:
        return self.results.comoving_radial_distance(z)

    def redshift_at_comoving_distance(self, chi: Any) -> Any:
        return self.results.redshift_at_comoving_radial_distance(chi)

    def transverse_comoving_distance(self, z: Any) -> Any:
        return self.results.angular_diameter_distance(z) * (1 + np.asarray(z))

    def luminosity_distance(self, z: Any) -> Any:
        return self.results.luminosity_distance(z)

    def angular_diameter_distance(self, z: Any) -> Any:
        return self.results.angular_diameter_distance(z)

    def differential_comoving_volume(self, z: Any) -> Any:
        return self.transverse_comoving_distance(z) ** 2 / self.hubble_parameter(z)
