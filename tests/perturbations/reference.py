"""Independent implementations of the matter power spectrum, to compare against.

Each class answers ``linear_matter_power_spectrum(k, z)`` and
``matter_power_spectrum(k, z)`` in cosmograd's units (1/Mpc, Mpc^3) and
broadcasting convention, so a cosmograd backend and a reference are compared
call for call. Build one from a cosmograd backend with ``from_perturbations``,
which reads the cosmology off the backend.

- :class:`JaxCosmoPerturbations`: jax-cosmo's Eisenstein-Hu spectrum.
- :class:`CAMBPerturbations`: CAMB's linear matter power spectrum.
- :class:`CloelibPerturbations`: cloelib's wrappers of the same emulators.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

from cosmograd import Perturbations
from cosmograd.perturbations.baccoemu import BaccoEmu
from cosmograd.perturbations.cosmopower import CosmoPower

from ..background.reference import camb_results
from ..cosmologies import Cosmology


def _scalar(value: Any) -> float:
    return float(np.asarray(value))


def _cosmology(perturbations: Perturbations) -> Cosmology:
    """The in-tree cosmology that every parameter of the backend is read from."""
    return cast(Cosmology, perturbations.background)


def sigma8(perturbations: Perturbations) -> Array:
    """RMS linear fluctuation in 8 Mpc/h spheres, by top-hat quadrature in ln k."""
    h = jnp.asarray(_cosmology(perturbations).H0) / 100
    ln_k = jnp.linspace(jnp.log(1e-5), jnp.log(50.0), 4000)
    k = jnp.exp(ln_k)
    x = k * 8 / h
    window = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
    integrand = k**3 * perturbations.linear_matter_power_spectrum(k, 0.0) * window**2
    return jnp.sqrt(jnp.trapezoid(integrand, ln_k) / (2 * jnp.pi**2))


def _elementwise(
    power_at_z: Callable[[float, np.ndarray], ArrayLike],
    k: ArrayLike,
    z: ArrayLike,
) -> Array:
    """Evaluate a spectrum defined at one redshift at a time, with ``(k, z)`` broadcasting.

    Parameters
    ----------
    power_at_z : callable
        ``power_at_z(z, k)`` returns the spectrum at a scalar redshift for a
        vector of wavenumbers.
    k, z : ArrayLike
        Concrete wavenumbers and redshifts, broadcast against each other.

    Returns
    -------
    Array
        The spectrum with the broadcast shape. Differentiable with respect to
        whatever ``power_at_z`` closes over.
    """
    k, z = np.broadcast_arrays(np.asarray(k, float), np.asarray(z, float))
    flat_k, flat_z = k.ravel(), z.ravel()
    redshifts, group = np.unique(flat_z, return_inverse=True)
    members = [np.flatnonzero(group == i) for i in range(len(redshifts))]
    pieces = [
        jnp.asarray(power_at_z(float(redshift), flat_k[index]))
        for redshift, index in zip(redshifts, members, strict=True)
    ]
    order = np.argsort(np.concatenate(members))
    return jnp.concatenate(pieces)[order].reshape(k.shape)


@dataclass(frozen=True)
class JaxCosmoPerturbations:
    """Eisenstein-Hu spectrum from jax-cosmo, normalised to a given ``sigma8``.

    jax-cosmo normalises through ``sigma8`` rather than ``A_s``, so the
    amplitude is taken from the backend under test. Only the shape and the
    redshift dependence are compared.
    """

    cosmology: Cosmology
    sigma8: ArrayLike

    @classmethod
    def from_perturbations(
        cls, perturbations: Perturbations, z: ArrayLike
    ) -> "JaxCosmoPerturbations":
        return cls(_cosmology(perturbations), sigma8(perturbations))

    def linear_matter_power_spectrum(self, k: ArrayLike, z: ArrayLike) -> Array:
        jc = pytest.importorskip("jax_cosmo")
        transfer = pytest.importorskip("jax_cosmo.transfer")
        cosmology = self.cosmology
        h = jnp.asarray(cosmology.H0) / 100
        cosmo = jc.Cosmology(
            Omega_c=jnp.asarray(cosmology.omega_m) - jnp.asarray(cosmology.omega_b),
            Omega_b=cosmology.omega_b,
            h=h,
            n_s=cosmology.n_s,
            sigma8=self.sigma8,
            Omega_k=0.0,
            w0=cosmology.w0,
            wa=cosmology.wa,
        )

        def at(z: float, k: np.ndarray) -> Array:
            # jax-cosmo works in h/Mpc and (Mpc/h)^3.
            pk = jc.power.linear_matter_power(
                cosmo,
                k / h,
                a=1 / (1 + z),
                transfer_fn=transfer.Eisenstein_Hu,
                type="eisenhu_osc",
            )
            return pk / h**3

        return _elementwise(at, k, z)

    matter_power_spectrum = linear_matter_power_spectrum


@dataclass(frozen=True)
class CAMBPerturbations:
    """CAMB's linear matter power spectrum (``mnu = 0``, PPF dark energy)."""

    interpolator: Any

    @classmethod
    def from_perturbations(
        cls, perturbations: Perturbations, z: ArrayLike
    ) -> "CAMBPerturbations":
        """Run CAMB for the backend's cosmology, storing power at ``z``."""
        results = camb_results(
            _cosmology(perturbations),
            power=True,
            kmax=2.0,
            zs=tuple(float(redshift) for redshift in np.unique(np.asarray(z))),
        )
        return cls(
            results.get_matter_power_interpolator(
                nonlinear=False, hubble_units=False, k_hunit=False
            )
        )

    def linear_matter_power_spectrum(self, k: ArrayLike, z: ArrayLike) -> Array:
        return _elementwise(lambda z, k: self.interpolator.P(z, k), k, z)

    matter_power_spectrum = linear_matter_power_spectrum


def _cloelib_background(cosmology: Cosmology) -> Any:
    jax_cosmology = pytest.importorskip("cloelib.cosmology.jax_cosmology")
    return jax_cosmology.JAXBackground(
        H0=_scalar(cosmology.H0),
        Omega_b0=_scalar(cosmology.omega_b),
        Omega_cdm0=_scalar(cosmology.omega_m) - _scalar(cosmology.omega_b),
        Omega_k0=0.0,
        As=math.exp(_scalar(cosmology.ln10as)) * 1e-10,
        ns=_scalar(cosmology.n_s),
        mnu=0.0,
        w0=_scalar(cosmology.w0),
        wa=_scalar(cosmology.wa),
        gamma_MG=0.55,
        N_mnu=0,
    )


@dataclass(frozen=True)
class CloelibPerturbations:
    """cloelib's wrapper of the CosmoPower or baccoemu emulators.

    cloelib builds each spectrum from its own background object for a fixed
    grid of redshifts, so the redshifts passed to the methods must be among
    those given to :meth:`from_perturbations`.
    """

    linear: Any
    full: Any
    """The nonlinear spectrum when the backend is nonlinear, otherwise ``linear``."""
    z: np.ndarray

    @classmethod
    def from_perturbations(
        cls, perturbations: Perturbations, z: ArrayLike
    ) -> "CloelibPerturbations":
        """Mirror a CosmoPower or BaccoEmu backend, linear or nonlinear.

        Raises
        ------
        ValueError
            From cloelib, when the cosmology is outside the training box.
        """
        z = np.asarray(z, float)
        background = _cloelib_background(_cosmology(perturbations))
        if isinstance(perturbations, CosmoPower):
            nonlinear = perturbations.nonlinear
            pytest.importorskip("cosmopower_jax")
            module = pytest.importorskip("cloelib.cosmology.cosmopower_jax_cosmology")
            emulators = module.CosmoPowerJAXLCDMPerturbations
            linear = emulators.Linear(background, z)
            full = emulators.NonLinear(background, linear, z) if nonlinear else linear
        elif isinstance(perturbations, BaccoEmu):
            nonlinear = perturbations.nonlinear
            pytest.importorskip("baccoemu.matter_powerspectrum")
            module = pytest.importorskip("cloelib.cosmology.baccoemu_cosmology")
            linear = module.BACCOemuLinearPerturbations(background, z)
            full = (
                module.BACCOemuNonLinearPerturbations(background, linear, z)
                if nonlinear
                else linear
            )
        else:
            raise TypeError(f"cloelib has no counterpart of {type(perturbations)}")
        return cls(linear, full, z)

    def _evaluate(self, reference: Any, k: ArrayLike, z: ArrayLike) -> Array:
        def at(z: float, k: np.ndarray) -> np.ndarray:
            (index,) = np.flatnonzero(np.isclose(self.z, z))
            # cloelib returns the spectrum with shape (z, k).
            return np.asarray(reference.matter_power_spectrum(self.z, k))[index]

        return _elementwise(at, k, z)

    def linear_matter_power_spectrum(self, k: ArrayLike, z: ArrayLike) -> Array:
        return self._evaluate(self.linear, k, z)

    def matter_power_spectrum(self, k: ArrayLike, z: ArrayLike) -> Array:
        return self._evaluate(self.full, k, z)
