"""Extrapolation of an emulated linear power spectrum beyond its trusted k range.

Kept free of optional-dependency imports so that importing :mod:`cosmograd`
never loads an emulator.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = ["KRangeExtrapolation", "extrapolate_log_pk"]

LogPk = Callable[[Array, Array], Array]


def extrapolate_log_pk(
    log_pk: LogPk,
    primordial: Callable[[Array], Array],
    k: ArrayLike,
    z: ArrayLike,
    k_min: ArrayLike,
    k_max: ArrayLike,
    slope_fraction: float = 0.9,
) -> Array:
    """:math:`\\ln P_\\mathrm{lin}(k, z)` continued outside ``[k_min, k_max]``.

    - Below ``k_min`` the transfer function is constant, so
      :math:`P \\propto k\\,\\mathcal{P}_\\mathcal{R}(k)`, anchored at ``k_min``.
    - Above ``k_max`` the log-log slope measured between
      ``slope_fraction * k_max`` and ``k_max`` is continued.

    ``log_pk`` only ever sees wavenumbers inside the range, so values and
    gradients stay finite whichever branch is selected.

    Parameters
    ----------
    log_pk : callable
        ``log_pk(k, z)``, the natural log of the spectrum, valid in range.
    primordial : callable
        ``primordial(k)``, the primordial curvature spectrum.
    k, z : ArrayLike
        Wavenumber and redshift, broadcast elementwise.
    k_min, k_max : ArrayLike
        Edges of the trusted range, in the units of ``k``. May be traced.
    slope_fraction : float, default 0.9
        The slope above ``k_max`` is measured over
        ``[slope_fraction * k_max, k_max]``; it must lie in (0, 1). Smaller
        values average over more of the range, larger ones follow the local
        slope more closely. Irrelevant for a pure power law.

    Returns
    -------
    Array
        :math:`\\ln P` with the broadcast shape of ``k`` and ``z``.
    """
    k = jnp.asarray(k, dtype=float)
    z = jnp.asarray(z, dtype=float)
    k_min = jnp.asarray(k_min, dtype=float)
    k_max = jnp.asarray(k_max, dtype=float)

    inside = log_pk(jnp.clip(k, k_min, k_max), z)

    log_edge = log_pk(k_max, z)
    slope = (log_edge - log_pk(slope_fraction * k_max, z)) / -jnp.log(slope_fraction)
    high = log_edge + slope * jnp.log(jnp.maximum(k, k_max) / k_max)

    k_low = jnp.minimum(k, k_min)
    low = (
        log_pk(k_min, z)
        + jnp.log(k_low / k_min)
        + jnp.log(primordial(k_low) / primordial(k_min))
    )

    return jnp.where(k < k_min, low, jnp.where(k > k_max, high, inside))


class KRangeExtrapolation:
    """Mixin giving a backend an opt-in extrapolated linear spectrum.

    The host provides ``k_min`` and ``k_max`` (1/Mpc), the static flags
    ``extrapolate`` and ``slope_fraction``, ``primordial_power_spectrum`` and
    ``_log_pk_at(k, z)``, the in-range :math:`\\ln P_\\mathrm{lin}`.
    """

    def _log_pk_extrapolated(self: Any, k: ArrayLike, z: ArrayLike) -> Array:
        """:math:`\\ln P_\\mathrm{lin}`, extrapolated when ``extrapolate`` is set."""
        k = jnp.asarray(k, dtype=float)
        z = jnp.asarray(z, dtype=float)
        if not self.extrapolate:
            return self._log_pk_at(k, z)
        return extrapolate_log_pk(
            self._log_pk_at,
            self.primordial_power_spectrum,
            k,
            z,
            self.k_min,
            self.k_max,
            self.slope_fraction,
        )
