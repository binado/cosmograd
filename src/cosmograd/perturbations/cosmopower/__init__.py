"""CosmoPower-JAX matter power spectrum. Needs the ``cosmopower`` extra.

:class:`CosmoPower` emulates the CLASS linear spectrum and
the HMcode nonlinear boost (probe ``mpk_boost``) of Spurio Mancini et al.
(2022), for a supplied flat :math:`\\Lambda\\mathrm{CDM}` cosmology.
The emulators are loaded on first use and never become pytree leaves.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from types import ModuleType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ...background import Background
from ...constants import C_KM_S
from .._extrapolation import KRangeExtrapolation
from ..growth import linear_growth, matter_domination_scale_factor

__all__ = ["CosmoPower", "CosmoPowerParameters", "load_cosmopower"]

K_PIVOT: float = 0.05
"""Pivot wavenumber of the primordial spectrum in 1/Mpc (CLASS default)."""

_Z_MAX = 5.0


def load_cosmopower() -> ModuleType:
    """Import ``cosmopower_jax`` on first use.

    Returns
    -------
    ModuleType
        The ``cosmopower_jax`` module.

    Raises
    ------
    ImportError
        If the ``cosmopower`` extra is not installed.
    """
    try:
        return importlib.import_module("cosmopower_jax")
    except ImportError as err:
        raise ImportError(
            "'cosmopower_jax' is required for this backend. "
            "Install it with `pip install 'cosmograd[cosmopower]'`."
        ) from err


@cache
def _emulator(probe: str) -> Any:
    """Load and cache one pre-trained CosmoPower-JAX network.

    Parameters
    ----------
    probe : str
        Probe name understood by ``cosmopower_jax``, e.g. ``"mpk_lin"``.

    Returns
    -------
    cosmopower_jax.cosmopower_jax.CosmoPowerJAX
        The emulator. Its weights are closed over as constants under
        :func:`jax.jit`.
    """
    load_cosmopower()
    module = importlib.import_module("cosmopower_jax.cosmopower_jax")
    return module.CosmoPowerJAX(probe=probe, verbose=False)


@cache
def _log_modes(probe: str) -> np.ndarray:
    """Natural log of the emulator's wavenumbers in 1/Mpc, as a NumPy array."""
    return np.log(np.asarray(_emulator(probe).modes, dtype=float))


class CosmoPowerParameters(NamedTuple):
    """What :class:`CosmoPower` needs and a background cannot provide."""

    omega_b: Real[ArrayLike, ""]
    """Baryon density parameter today, :math:`\\Omega_b`."""
    n_s: Real[ArrayLike, ""]
    """Scalar spectral index."""
    ln10as: Real[ArrayLike, ""]
    """:math:`\\ln(10^{10} A_s)`, with :math:`A_s` at the pivot 0.05/Mpc."""


def _attributes(background: Any) -> CosmoPowerParameters:
    """Read the parameters from the attributes of a native cosmology."""
    return CosmoPowerParameters(
        omega_b=background.omega_b, n_s=background.n_s, ln10as=background.ln10as
    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["background", "c_min", "eta_0"],
    meta_fields=[
        "parameters",
        "k_growth",
        "nonlinear",
        "extrapolate",
        "extrapolation_slope_fraction",
    ],
)
@dataclass(frozen=True)
class CosmoPower[B: Background](KRangeExtrapolation):
    """Flat :math:`\\Lambda\\mathrm{CDM}` spectra from the CosmoPower-JAX emulators.

    The linear spectrum comes from the ``mpk_lin`` network and the nonlinear
    one multiplies it by the HMcode boost of ``mpk_boost``. Both are emulated
    on a fixed grid of wavenumbers and interpolated linearly in
    :math:`\\ln k` and :math:`\\ln P`. The emulators already work in 1/Mpc and
    :math:`\\mathrm{Mpc}^3`, so no unit conversion is needed.

    Growth is read off the linear spectrum,
    :math:`D(z, k) = \\sqrt{P(k, z) / P(k, 0)}` and
    :math:`f = -\\tfrac{1}{2}(1 + z)\\,\\partial_z \\ln P(k, z)` by automatic
    differentiation. Without ``k`` both are evaluated at ``k_growth``, where
    the emulated growth is scale independent.

    Parameters
    ----------
    background : Background
        The cosmology, retained by identity. Its Hubble constant and matter
        density today are emulator inputs (baryons plus cold dark matter;
        neutrinos are massless). Must be flat :math:`\\Lambda\\mathrm{CDM}`:
        the emulators know no other expansion history.
    parameters : callable, optional
        ``parameters(background)``, returning the
        :class:`CosmoPowerParameters`. The default reads the attributes
        ``omega_b``, ``n_s`` and ``ln10as`` of the background, which the
        native cosmologies have. Static; define it at module level so that
        :func:`jax.jit` does not recompile for every instance.
    c_min : ArrayLike, default 3.13
        HMcode halo concentration amplitude. Only affects ``nonlinear=True``.
    eta_0 : ArrayLike, default 0.603
        HMcode halo bloating parameter. Only affects ``nonlinear=True``.
    k_growth : float, default 0.01
        Wavenumber in 1/Mpc at which growth is evaluated when ``k`` is
        ``None``. Static.

    nonlinear : bool, default False
        Apply the HMcode boost to the matter power spectrum. Static.
    extrapolate : bool, default False
        Continue the linear spectrum outside ``[k_min, k_max]``, see Notes.
        Static.
    extrapolation_slope_fraction : float, default 0.9
        The log-log slope above ``k_max`` is measured between
        ``extrapolation_slope_fraction * k_max`` and ``k_max``; must lie in (0, 1). Static.

    Notes
    -----
    The emulators are only accurate inside their training ranges, which no
    method checks:

    - :math:`\\omega_b = \\Omega_b h^2 \\in [0.01875, 0.02625]`,
      :math:`\\omega_c = (\\Omega_m - \\Omega_b) h^2 \\in [0.05, 0.255]`,
      :math:`h \\in [0.64, 0.82]`, :math:`n_s \\in [0.84, 1.1]`,
      :math:`\\ln(10^{10} A_s) \\in [1.61, 3.91]`,
      :math:`c_\\mathrm{min} \\in [2, 4]`, :math:`\\eta_0 \\in [0.5, 1]`;
    - :math:`0 \\le z \\le 5`;
    - :math:`10^{-5} \\le k \\le 9.8` 1/Mpc (:attr:`k_min`, :attr:`k_max`).
      By default the spectrum is held at its value at the nearest end.

    Above ``z = 5``, linear power uses the spectrum at ``z = 5`` scaled by
    :math:`[D_{\\mathrm{ODE}}(z) / D_{\\mathrm{ODE}}(5)]^2`. The ODE uses the
    supplied background and starts at the scale factor selected by
    :func:`~cosmograd.perturbations.growth.matter_domination_scale_factor`.
    Redshifts before that epoch return NaN. The power and growth factor are
    continuous at the boundary, where the growth rate takes the emulator
    value and may jump.

    With ``nonlinear=True``, :meth:`matter_power_spectrum` evaluates both
    emulators at the requested redshift. Its predictions above ``z = 5``
    are outside their training range. The explicit linear power and growth
    methods still use the ODE continuation.

    With ``extrapolate=True`` the linear spectrum, and the growth functions
    read off it, are continued smoothly instead. Below ``k_min`` the transfer
    function is taken constant, :math:`P \\propto k\\,\\mathcal{P}_\\mathcal{R}(k)`;
    above ``k_max`` the log-log slope measured over
    ``[extrapolation_slope_fraction * k_max, k_max]`` continues. The HMcode boost stays
    clamped to its grid.

    The emulators were trained on CLASS with massless neutrinos, matching the
    radiation content of :class:`~cosmograd.FlatLCDM` at its defaults. Custom
    radiation settings are retained for background calculations but do not
    affect the emulated spectra. They do affect the ODE continuation.

    References
    ----------
    A. Spurio Mancini et al., MNRAS 511, 1771 (2022), arXiv:2106.03846.
    D. Piras and A. Spurio Mancini, OJAp 6, 20 (2023), arXiv:2305.06347.
    """

    # Tells jaxtyping's import hook not to check the generated ``__init__``;
    # see :class:`~cosmograd.FlatLCDM`.
    __no_type_check__ = True

    background: B
    parameters: Callable[[B], CosmoPowerParameters] = _attributes
    c_min: Real[ArrayLike, ...] = 3.13
    eta_0: Real[ArrayLike, ...] = 0.603
    k_growth: float = 0.01
    nonlinear: bool = False
    extrapolate: bool = False
    extrapolation_slope_fraction: float = 0.9

    @property
    def k_min(self) -> Float[Array, ""]:
        """Smallest wavenumber of the emulator grid, in 1/Mpc."""
        return jnp.asarray(np.exp(_log_modes("mpk_lin")[0]))

    @property
    def k_max(self) -> Float[Array, ""]:
        """Largest wavenumber of the emulator grid, in 1/Mpc."""
        return jnp.asarray(np.exp(_log_modes("mpk_lin")[-1]))

    def _emulator_inputs(self, z: Array) -> dict[str, Array]:
        """Translate the parameters to CosmoPower's names, one row per redshift.

        Parameters
        ----------
        z : Array
            Redshifts, flattened, shape ``(n,)``.

        Returns
        -------
        dict
            Every input name used by ``mpk_lin`` and ``mpk_boost``, each an
            array of shape ``(n,)``.
        """
        parameters = self.parameters(self.background)
        h = self.background.hubble_constant * C_KM_S / 100
        h2 = h**2
        omega_m = self.background.matter_density_parameter(0.0)
        omega_b = jnp.asarray(parameters.omega_b)
        params = {
            "omega_b": omega_b * h2,
            "omega_cdm": (omega_m - omega_b) * h2,
            "h": h,
            "n_s": parameters.n_s,
            "ln10^{10}A_s": parameters.ln10as,
            "cmin": self.c_min,
            "eta_0": self.eta_0,
        }
        return {"z": z} | {
            name: jnp.broadcast_to(jnp.asarray(value), z.shape)
            for name, value in params.items()
        }

    def _log_spectrum(self, probe: str, z: Array) -> Array:
        """Natural log of one emulator's output on its own wavenumber grid.

        Parameters
        ----------
        probe : str
            ``"mpk_lin"`` (spectrum in Mpc^3) or ``"mpk_boost"`` (ratio).
        z : Array
            Redshifts of any shape.

        Returns
        -------
        Array
            Shape ``z.shape + (n_modes,)``.
        """
        emulator = _emulator(probe)
        flat = z.reshape(-1)
        inputs = self._emulator_inputs(flat)
        x = jnp.stack([inputs[name] for name in emulator.parameters], axis=-1)
        out = emulator.predict(x).reshape(flat.shape[0], -1)
        return jnp.log(out).reshape(*z.shape, -1)

    def _interpolate(self, probe: str, k: Array, z: Array) -> Array:
        """Table of one probe interpolated at ``k``, clamped to the grid.

        The emulator runs once per element of ``z`` (before broadcasting), so
        an outer-product grid ``k[:, None], z[None, :]`` costs one network
        pass per redshift.
        """
        table = self._log_spectrum(probe, z)
        log_modes = jnp.asarray(_log_modes("mpk_lin"))
        log_k = jnp.clip(jnp.log(k), log_modes[0], log_modes[-1])
        upper = jnp.clip(jnp.searchsorted(log_modes, log_k), 1, log_modes.size - 1)
        lower = upper - 1
        weight = (log_k - log_modes[lower]) / (log_modes[upper] - log_modes[lower])

        shape = jnp.broadcast_shapes(k.shape, z.shape)
        table = jnp.broadcast_to(table, (*shape, log_modes.size))

        def gather(index: Array) -> Array:
            index = jnp.broadcast_to(index, shape)[..., None]
            return jnp.take_along_axis(table, index, axis=-1)[..., 0]

        return (1 - weight) * gather(lower) + weight * gather(upper)

    def _log_pk_at(self, k: Array, z: Array) -> Array:
        return self._interpolate("mpk_lin", k, z)

    def _log_growth_correction(self, z: Array, above: Array) -> Array:
        """Return the log-power correction for redshifts above the training range."""
        a_initial = matter_domination_scale_factor(self.background)
        # Use the solver's log-scale-factor coordinates at the initial endpoint.
        supported = -jnp.log1p(z) >= jnp.log(a_initial)
        z_solve = jnp.where(above & supported, z, _Z_MAX)
        d, _ = linear_growth(
            self.background,
            jnp.append(z_solve.ravel(), _Z_MAX),
            normalize=False,
            a_initial=a_initial,
        )
        correction = 2 * (jnp.log(d[:-1].reshape(z.shape)) - jnp.log(d[-1]))
        return jnp.where(above, jnp.where(supported, correction, jnp.nan), 0.0)

    def _log_power(self, k: ArrayLike, z: ArrayLike, nonlinear: bool) -> Array:
        """:math:`\\ln P(k, z)`: linear spectrum, extrapolated if enabled, plus boost."""
        k = jnp.asarray(k, dtype=float)
        z = jnp.asarray(z, dtype=float)
        if nonlinear:
            return self._log_pk_extrapolated(k, z) + self._interpolate(
                "mpk_boost", k, z
            )

        above = z > _Z_MAX
        z_emulator = jnp.where(above, _Z_MAX, z)
        correction = jax.lax.cond(
            jnp.any(above),
            lambda: self._log_growth_correction(z, above),
            lambda: jnp.zeros_like(z),
        )
        return self._log_pk_extrapolated(k, z_emulator) + correction

    def matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Matter power spectrum :math:`P(k, z)`.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive.
        z : ArrayLike
            Non-negative redshift. Broadcast elementwise against ``k``.
            The nonlinear emulator prediction is trained through ``z = 5``.

        Returns
        -------
        Array
            :math:`P(k, z)` in :math:`\\mathrm{Mpc}^3`, with the broadcast
            shape of ``k`` and ``z``.
        """
        return jnp.exp(self._log_power(k, z, self.nonlinear))

    def linear_matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Linear matter power spectrum, whatever ``nonlinear`` is.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive.
        z : ArrayLike
            Non-negative redshift. Broadcast elementwise against ``k``.
            Above ``z = 5``, use the ODE continuation described on the class.

        Returns
        -------
        Array
            :math:`P_\\mathrm{lin}(k, z)` in :math:`\\mathrm{Mpc}^3`, with the
            broadcast shape of ``k`` and ``z``.
        """
        return jnp.exp(self._log_power(k, z, False))

    def _growth_k(self, k: ArrayLike | None) -> Array:
        """Wavenumber to evaluate growth at, broadcast against ``z``."""
        return jnp.asarray(self.k_growth if k is None else k, dtype=float)

    def growth_function(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Linear growth :math:`D = \\sqrt{P(k, z) / P(k, 0)}`, normalised at z = 0.

        Parameters
        ----------
        z : ArrayLike
            Non-negative redshift, with ODE continuation above ``z = 5``.
        k : ArrayLike, optional
            Wavenumber in 1/Mpc. ``None`` uses ``k_growth``.

        Returns
        -------
        Array
            Dimensionless. Shape of ``z`` when ``k`` is ``None``, otherwise the
            broadcast shape of ``k`` and ``z``.
        """
        z = jnp.asarray(z, dtype=float)
        k = self._growth_k(k)
        log_ratio = self._log_power(k, z, False) - self._log_power(k, 0.0, False)
        return jnp.exp(0.5 * log_ratio)

    def growth_rate(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Linear growth rate :math:`f = -\\tfrac{1}{2}(1 + z)\\,\\partial_z \\ln P`.

        Parameters
        ----------
        z : ArrayLike
            Non-negative redshift, with ODE continuation above ``z = 5``.
        k : ArrayLike, optional
            Wavenumber in 1/Mpc. ``None`` uses ``k_growth``.

        Returns
        -------
        Array
            Dimensionless. Shape of ``z`` when ``k`` is ``None``, otherwise the
            broadcast shape of ``k`` and ``z``.
        """
        z = jnp.asarray(z, dtype=float)
        k = self._growth_k(k)
        # Each output depends on exactly one element of z, so a JVP with unit
        # tangents is the elementwise derivative.
        log_pk, dlnp_dz = jax.jvp(
            lambda z_: self._log_power(k, z_, False), (z,), (jnp.ones_like(z),)
        )
        return jnp.where(jnp.isnan(log_pk), jnp.nan, -0.5 * (1 + z) * dlnp_dz)

    def primordial_power_spectrum(
        self,
        k: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Power-law curvature spectrum :math:`A_s (k / k_*)^{n_s - 1}`.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive.

        Returns
        -------
        Array
            :math:`\\mathcal{P}_\\mathcal{R}(k)`, dimensionless, with the shape
            of ``k``. The pivot is ``K_PIVOT`` = 0.05/Mpc.
        """
        parameters = self.parameters(self.background)
        a_s = 1e-10 * jnp.exp(jnp.asarray(parameters.ln10as))
        return a_s * (jnp.asarray(k, dtype=float) / K_PIVOT) ** (
            jnp.asarray(parameters.n_s) - 1
        )
