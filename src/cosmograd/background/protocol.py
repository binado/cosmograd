"""Protocol for homogeneous expansion history and distances.

The protocols list behaviour only. Which input parameters exist is up to each
implementation, so consumers should never read attributes such as ``cosmo.h``.

Conventions shared by every implementation:

- Distances in Mpc, wavenumbers in 1/Mpc, the Hubble rate in 1/Mpc (:math:`c = 1`)
  and power spectra in :math:`\\mathrm{Mpc}^3`.
- ``(k, z)`` order whenever both are positional.
- One cosmology per object; ``k`` and ``z`` broadcast elementwise. Batch over
  cosmologies with :func:`jax.vmap`.
- A cosmology is a pytree whose only leaves are its parameters. A
  :class:`typing.Protocol` cannot express this, so it is checked by
  :mod:`cosmograd.testing` instead.
"""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

__all__ = ["Background"]


@runtime_checkable
class Background(Protocol):
    """Homogeneous expansion history: Hubble rate and distances.

    Implementations must provide :attr:`hubble_constant`,
    :meth:`hubble_parameter`, :meth:`matter_density_parameter`,
    :meth:`comoving_distance` and :meth:`redshift_at_comoving_distance`.
    The remaining methods are written in terms of those. A class gets them by
    inheriting from ``Background`` explicitly, which is the documented way to
    implement a backend; a class that only matches structurally must write
    them itself.

    The derived distances all go through :meth:`transverse_comoving_distance`,
    so a curved model overrides that one method and the others stay correct.

    Unless an implementation says otherwise, redshift arguments are expected to
    be non-negative and inside the range the implementation documents. No
    method checks its inputs.
    """

    @property
    def hubble_constant(self) -> Float[Array, ""]:
        """Hubble constant :math:`H_0`.

        Returns
        -------
        Array
            :math:`H_0` in 1/Mpc (:math:`c = 1`), scalar.
        """
        ...

    def hubble_parameter(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Hubble rate :math:`H(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`H(z)` in 1/Mpc, with the shape of ``z``.
        """
        ...

    def matter_density_parameter(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Matter density parameter :math:`\\Omega_m(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`\\Omega_m(z) = \\rho_m(z) / \\rho_\\mathrm{crit}(z)`,
            dimensionless, with the shape of ``z``.
        """
        ...

    def comoving_distance(self, z: Real[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
        """Line-of-sight comoving distance :math:`\\chi(z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`\\chi(z) = \\int_0^z \\mathrm{d}z' / H(z')` in Mpc, with the
            shape of ``z``.
        """
        ...

    def redshift_at_comoving_distance(
        self,
        chi: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Inverse of :meth:`comoving_distance`.

        Parameters
        ----------
        chi : ArrayLike
            Comoving distance in Mpc, non-negative.

        Returns
        -------
        Array
            Redshift :math:`z(\\chi)`, with the shape of ``chi``.
        """
        ...

    def transverse_comoving_distance(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Transverse comoving distance :math:`D_M(z)`.

        The default assumes a flat universe, :math:`D_M = \\chi`. Curved models
        override this method.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`D_M(z)` in Mpc, with the shape of ``z``.
        """
        return self.comoving_distance(z)

    def luminosity_distance(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Luminosity distance :math:`d_L = (1 + z) D_M`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`d_L(z)` in Mpc, with the shape of ``z``.
        """
        return (1 + jnp.asarray(z)) * self.transverse_comoving_distance(z)

    def angular_diameter_distance(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Angular diameter distance :math:`d_A = D_M / (1 + z)`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`d_A(z)` in Mpc, with the shape of ``z``.
        """
        return self.transverse_comoving_distance(z) / (1 + jnp.asarray(z))

    def differential_comoving_volume(
        self,
        z: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Comoving volume per unit redshift and solid angle.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.

        Returns
        -------
        Array
            :math:`\\mathrm{d}V / \\mathrm{d}z\\,\\mathrm{d}\\Omega = D_M^2 / H`
            in :math:`\\mathrm{Mpc}^3 / \\mathrm{sr}`, with the shape of ``z``.
        """
        return self.transverse_comoving_distance(z) ** 2 / self.hubble_parameter(z)
