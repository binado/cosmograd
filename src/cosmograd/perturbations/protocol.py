"""Protocol for linear growth and power spectra.

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

from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..background import Background

__all__ = ["Perturbations"]


@runtime_checkable
class Perturbations(Protocol):
    """Linear growth and power spectra on top of a :class:`Background`.

    The contract does not say whether the matter power spectrum factorises
    into a function of ``k`` times a function of ``z``. Consumers that need the
    factorised form check for it themselves.
    """

    @property
    def background(self) -> Background:
        """The background to use for distances alongside these perturbations.

        It need not be the exact expansion history behind the spectra: an
        emulator fixes its own, and an implementation may leave out components
        its background class cannot represent. Each implementation documents
        how the two relate.

        Returns
        -------
        Background
            May be ``self`` when one object implements both protocols.
        """
        ...

    def growth_function(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Linear growth function :math:`D`, normalised to 1 at :math:`z = 0`.

        With scale dependence it is defined through
        :meth:`linear_matter_power_spectrum`,

        .. math::

            D(z, k) = \\sqrt{\\frac{P_\\mathrm{lin}(k, z)}{P_\\mathrm{lin}(k, 0)}}.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.
        k : ArrayLike, optional
            Wavenumber in 1/Mpc. ``None`` means the large-scale limit.
            Scale-independent implementations accept ``k`` and broadcast the
            same value over it.

        Returns
        -------
        Array
            Dimensionless. Shape of ``z`` when ``k`` is ``None``, otherwise the
            broadcast shape of ``k`` and ``z``.
        """
        ...

    def growth_rate(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Linear growth rate :math:`f = \\mathrm{d}\\ln D / \\mathrm{d}\\ln a`.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.
        k : ArrayLike, optional
            Wavenumber in 1/Mpc. ``None`` means the large-scale limit.

        Returns
        -------
        Array
            Dimensionless. Shape of ``z`` when ``k`` is ``None``, otherwise the
            broadcast shape of ``k`` and ``z``.
        """
        ...

    def matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Matter power spectrum :math:`P(k, z)`.

        Linear or nonlinear power is selected when constructing the instance.
        :meth:`linear_matter_power_spectrum` is linear either way.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc (never h/Mpc), positive.
        z : ArrayLike
            Redshift, non-negative. Broadcast elementwise against ``k``.

        Returns
        -------
        Array
            :math:`P(k, z)` in :math:`\\mathrm{Mpc}^3`, with the broadcast
            shape of ``k`` and ``z``.
        """
        ...

    def linear_matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Linear matter power spectrum :math:`P_\\mathrm{lin}(k, z)`.

        Independent of whether the instance was constructed for linear or
        nonlinear power. It is the spectrum that defines
        :meth:`growth_function`.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc (never h/Mpc), positive.
        z : ArrayLike
            Redshift, non-negative. Broadcast elementwise against ``k``.

        Returns
        -------
        Array
            :math:`P_\\mathrm{lin}(k, z)` in :math:`\\mathrm{Mpc}^3`, with the
            broadcast shape of ``k`` and ``z``.
        """
        ...

    def primordial_power_spectrum(
        self,
        k: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Dimensionless primordial curvature power spectrum.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive.

        Returns
        -------
        Array
            :math:`\\mathcal{P}_\\mathcal{R}(k)`, dimensionless, with the shape
            of ``k``.
        """
        ...
