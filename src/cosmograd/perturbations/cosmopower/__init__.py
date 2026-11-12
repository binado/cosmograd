"""CosmoPower-JAX matter power spectrum. Needs the ``cosmopower`` extra.

:class:`CosmoPower` emulates the CLASS linear spectrum (probe ``mpk_lin``) and
the HMcode nonlinear boost (probe ``mpk_boost``) of Spurio Mancini et al.
(2022), and pairs them with a native :class:`~cosmograd.FlatLCDM` background.
The emulators are loaded on first use and never become pytree leaves.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache, partial
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ...background import FlatLCDM
from ..protocol import Perturbations

__all__ = ["CosmoPower", "load_cosmopower"]

K_PIVOT: float = 0.05
"""Pivot wavenumber of the primordial spectrum in 1/Mpc (CLASS default)."""


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


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["h", "omega_m", "omega_b", "n_s", "ln10as", "c_min", "eta_0"],
    meta_fields=["k_growth", "nonlinear"],
)
@dataclass(frozen=True)
class CosmoPower(Perturbations):
    """Flat :math:`\\Lambda\\mathrm{CDM}` spectra from the CosmoPower-JAX emulators.

    The linear spectrum comes from the ``mpk_lin`` network and the nonlinear
    one multiplies it by the HMcode boost of ``mpk_boost``. Both are emulated
    on a fixed grid of wavenumbers and interpolated linearly in
    :math:`\\ln k` and :math:`\\ln P`. The emulators already work in 1/Mpc and
    :math:`\\mathrm{Mpc}^3`, so no unit conversion is needed.

    Growth is read off the emulated linear spectrum,
    :math:`D(z, k) = \\sqrt{P(k, z) / P(k, 0)}` and
    :math:`f = -\\tfrac{1}{2}(1 + z)\\,\\partial_z \\ln P(k, z)` by automatic
    differentiation. Without ``k`` both are evaluated at ``k_growth``, where
    the emulated growth is scale independent.

    The instance is a pytree whose leaves are the parameter fields;
    ``k_growth`` and ``nonlinear`` are static. The emulator networks are
    module-level constants, never leaves.
    Methods operate on one cosmology with scalar parameters; batch with
    :func:`jax.vmap`.

    Parameters
    ----------
    h : ArrayLike
        Dimensionless Hubble constant, :math:`H_0 = 100\\,h` km/s/Mpc.
    omega_m : ArrayLike
        Matter density parameter today, :math:`\\Omega_m` (baryons plus cold
        dark matter; neutrinos are massless).
    omega_b : ArrayLike
        Baryon density parameter today, :math:`\\Omega_b`.
    n_s : ArrayLike
        Scalar spectral index.
    ln10as : ArrayLike
        :math:`\\ln(10^{10} A_s)`, with :math:`A_s` the primordial amplitude
        at the pivot ``K_PIVOT`` = 0.05/Mpc.
    c_min : ArrayLike, default 3.13
        HMcode halo concentration amplitude. Only affects ``nonlinear=True``.
    eta_0 : ArrayLike, default 0.603
        HMcode halo bloating parameter. Only affects ``nonlinear=True``.
    k_growth : float, default 0.01
        Wavenumber in 1/Mpc at which growth is evaluated when ``k`` is
        ``None``. Static.

    nonlinear : bool, default False
        Apply the HMcode boost to the matter power spectrum. Static.

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
    - :math:`10^{-5} \\le k \\le 9.8` 1/Mpc. Outside this range the
      spectrum is held at its value at the nearest end.

    The emulators were trained on CLASS with massless neutrinos, matching the
    radiation content of :class:`~cosmograd.FlatLCDM` at its defaults.

    References
    ----------
    A. Spurio Mancini et al., MNRAS 511, 1771 (2022), arXiv:2106.03846.
    D. Piras and A. Spurio Mancini, OJAp 6, 20 (2023), arXiv:2305.06347.
    """

    # Tells jaxtyping's import hook not to check the generated ``__init__``;
    # see :class:`~cosmograd.FlatLCDM`.
    __no_type_check__ = True

    h: Real[ArrayLike, ...]
    omega_m: Real[ArrayLike, ...]
    omega_b: Real[ArrayLike, ...]
    n_s: Real[ArrayLike, ...]
    ln10as: Real[ArrayLike, ...]
    c_min: Real[ArrayLike, ...] = 3.13
    eta_0: Real[ArrayLike, ...] = 0.603
    k_growth: float = 0.01
    nonlinear: bool = False

    @property
    def background(self) -> FlatLCDM:
        """Flat :math:`\\Lambda\\mathrm{CDM}` background with the same ``h`` and ``omega_m``.

        Returns
        -------
        FlatLCDM
            Background with default radiation content.
        """
        return FlatLCDM(H0=100 * jnp.asarray(self.h), omega_m=self.omega_m)

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
        h2 = jnp.asarray(self.h) ** 2
        params = {
            "omega_b": jnp.asarray(self.omega_b) * h2,
            "omega_cdm": (jnp.asarray(self.omega_m) - jnp.asarray(self.omega_b)) * h2,
            "h": self.h,
            "n_s": self.n_s,
            "ln10^{10}A_s": self.ln10as,
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

    def _log_power(self, k: ArrayLike, z: ArrayLike, nonlinear: bool) -> Array:
        """:math:`\\ln P(k, z)` interpolated from the emulator grid.

        The emulator runs once per element of ``z`` (before broadcasting), so
        an outer-product grid ``k[:, None], z[None, :]`` costs one network
        pass per redshift.
        """
        k = jnp.asarray(k, dtype=float)
        z = jnp.asarray(z, dtype=float)
        table = self._log_spectrum("mpk_lin", z)
        if nonlinear:
            table = table + self._log_spectrum("mpk_boost", z)

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
            Redshift in ``[0, 5]``. Broadcast elementwise against ``k``.

        Returns
        -------
        Array
            :math:`P(k, z)` in :math:`\\mathrm{Mpc}^3`, with the broadcast
            shape of ``k`` and ``z``.
        """
        return jnp.exp(self._log_power(k, z, self.nonlinear))

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
            Redshift in ``[0, 5]``.
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
            Redshift in ``[0, 5]``.
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
        _, dlnp_dz = jax.jvp(
            lambda z_: self._log_power(k, z_, False), (z,), (jnp.ones_like(z),)
        )
        return -0.5 * (1 + z) * dlnp_dz

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
        a_s = 1e-10 * jnp.exp(jnp.asarray(self.ln10as))
        return a_s * (jnp.asarray(k, dtype=float) / K_PIVOT) ** (self.n_s - 1)
