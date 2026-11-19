"""The bacco emulators behind the :class:`~cosmograd.Perturbations` protocol."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
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
from ._emulator import (
    LINEAR_KEYS,
    NONLINEAR_KEYS,
    SIGMA8_KEYS,
    Emulator,
    ModelName,
    load_emulator,
)
from ._power import (
    K_PIVOT,
    NU_DENSITY,
    amplitudes,
    interp_extrap,
    linear_log_pk,
    nonlinear_boost,
)

__all__ = ["BaccoEmu", "BaccoEmuParameters"]

# Least-squares slope over 5 points spaced by 0.03 in ln a (the centre has
# weight zero). Autodiff of the piecewise-linear ReLU networks is ~5x noisier.
_GROWTH_STENCIL = 0.03 * np.array([-2.0, -1.0, 1.0, 2.0])
_GROWTH_WEIGHTS = _GROWTH_STENCIL / np.sum(_GROWTH_STENCIL**2)

_Table = Callable[[Float[Array, ""]], Float[Array, " grid"]]

# Parameter transformations outside the networks follow ``jax_enable_x64``.
_float = partial(jnp.asarray, dtype=float)


class BaccoEmuParameters(NamedTuple):
    """What :class:`BaccoEmu` needs and a background cannot provide."""

    omega_b: Real[ArrayLike, ""]
    """Baryon density parameter today, :math:`\\Omega_b`."""
    n_s: Real[ArrayLike, ""]
    """Scalar spectral index."""
    ln10as: Real[ArrayLike, ""] | None = None
    """:math:`\\ln(10^{10} A_s)`, with :math:`A_s` at the pivot 0.05/Mpc.

    Only read when the backend has no ``sigma8_cold``.
    """
    w0: Real[ArrayLike, ""] = -1.0
    """Dark energy equation of state today."""
    wa: Real[ArrayLike, ""] = 0.0
    """Its derivative with respect to :math:`-a`."""


def _attributes(background: Any) -> BaccoEmuParameters:
    """Read the parameters from the attributes of a native cosmology."""
    return BaccoEmuParameters(
        omega_b=background.omega_b,
        n_s=background.n_s,
        ln10as=background.ln10as,
        w0=background.w0,
        wa=background.wa,
    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["background", "sigma8_cold", "neutrino_mass"],
    meta_fields=[
        "parameters",
        "model_name",
        "cold",
        "nonlinear",
        "extrapolate",
        "slope_fraction",
    ],
)
@dataclass(frozen=True)
class BaccoEmu[B: Background](KRangeExtrapolation):
    """Linear and nonlinear matter power spectra from the bacco emulators.

    A differentiable JAX port of ``baccoemu.Matter_powerspectrum`` (Aricò et
    al. 2021; Angulo et al. 2021) with its default settings: no-wiggles BAO
    smearing and no baryonic boost. The ``bacco`` extra provides the native
    JAX networks, evaluated in float32. Models are cached per ``model_name``
    and never become pytree leaves.

    ``k`` and ``z`` broadcast elementwise. The networks run once per element
    of ``z`` (before broadcasting), so pass ``z`` with the smallest shape that
    broadcasts against ``k``.

    Parameters
    ----------
    background : Background
        The cosmology, retained by identity. Its Hubble constant gives the
        emulator's ``hubble`` and its matter density today, which includes
        massive neutrinos, gives
        :math:`\\Omega_{cb} = \\Omega_m - \\Omega_\\nu`. Must be flat with CPL
        dark energy: the emulators know no other expansion history.
    parameters : callable, optional
        ``parameters(background)``, returning the :class:`BaccoEmuParameters`.
        The default reads the attributes ``omega_b``, ``n_s``, ``ln10as``,
        ``w0`` and ``wa`` of the background, which the native flat cosmologies
        have. Static; define it at module level so that :func:`jax.jit` does
        not recompile for every instance.
    sigma8_cold : ArrayLike, optional
        RMS linear cold-matter fluctuation in spheres of 8 Mpc/h at
        :math:`z = 0`. When given it sets the amplitude and the parameters'
        ``ln10as`` is ignored; otherwise ``ln10as`` sets it.
    neutrino_mass : ArrayLike, default 0.0
        Sum of the neutrino masses in eV,
        :math:`\\Omega_\\nu h^2 = M_\\nu / 93.14`.
    model_name : {"Angulo2021", "Arico2023"}, default "Angulo2021"
        Nonlinear emulator. Static.
    cold : bool, default False
        Return the cold-matter (CDM + baryons) spectrum instead of the total
        matter one, and cold-matter growth. Static.
    nonlinear : bool, default False
        Return the nonlinear matter power spectrum. Static.
    extrapolate : bool, default False
        Continue the linear spectrum outside ``[k_min, k_max]``, see Notes.
        Static.
    slope_fraction : float, default 0.9
        The log-log slope above ``k_max`` is measured between
        ``slope_fraction * k_max`` and ``k_max``; must lie in (0, 1). Static.

    Notes
    -----
    The emulators are trained on a bounded parameter box and k and z ranges,
    and they extrapolate silently outside it; call :meth:`check_bounds`
    eagerly. In ``baccoemu`` 2.3 the linear networks cover
    :math:`10^{-4} \\le k / (h\\,\\mathrm{Mpc}^{-1}) \\le 50` and
    :math:`z \\le 9`; the ``"Angulo2021"`` nonlinear network covers
    :math:`k \\le 4.9\\,h\\,\\mathrm{Mpc}^{-1}` and :math:`z \\le 1.5`, and
    ``"Arico2023"`` :math:`k \\le 9.8\\,h\\,\\mathrm{Mpc}^{-1}` and
    :math:`z \\le 2.6`. By default the spectrum outside the linear k range,
    :attr:`k_min` to :attr:`k_max` (which scale with :math:`h`), is the
    emulator's own power-law extrapolation; below the nonlinear grid the
    nonlinear spectrum equals the linear one.

    With ``extrapolate=True`` the linear spectrum, and growth at an explicit
    ``k``, follow a documented policy instead. Below ``k_min`` the transfer
    function is taken constant,
    :math:`P \\propto k\\,\\mathcal{P}_\\mathcal{R}(k)`; above ``k_max`` the
    log-log slope measured over ``[slope_fraction * k_max, k_max]`` continues.
    The nonlinear boost keeps its own behaviour and multiplies the extrapolated
    linear spectrum.

    The emulated expansion history is not exactly that of
    :attr:`background`, which treats massive neutrinos as matter at every
    redshift and keeps its own radiation content.

    The growth functions come from the linear emulator. Its ReLU networks
    are piecewise linear in :math:`a`, so :meth:`growth_rate` is a finite
    difference rather than a derivative, accurate to about a per cent
    against the growth ODE in :math:`\\Lambda\\mathrm{CDM}`. At
    :math:`z \\approx 0` it extrapolates the network slightly past
    :math:`a = 1`.
    """

    # See FlatLCDM: keep jaxtyping from checking the generated ``__init__``,
    # which JAX calls with placeholder leaves during transforms.
    __no_type_check__ = True

    background: B
    parameters: Callable[[B], BaccoEmuParameters] = _attributes
    sigma8_cold: Real[ArrayLike, "..."] | None = None
    neutrino_mass: Real[ArrayLike, "..."] = 0.0
    model_name: ModelName = "Angulo2021"
    cold: bool = False
    nonlinear: bool = False
    extrapolate: bool = False
    slope_fraction: float = 0.9

    @property
    def k_min(self) -> Float[Array, ""]:
        """Smallest wavenumber of the linear emulator in 1/Mpc, :math:`10^{-4}\\,h`."""
        return jnp.asarray(self._emulator.k_lin[0]) * self._h

    @property
    def k_max(self) -> Float[Array, ""]:
        """Largest wavenumber of the linear emulator in 1/Mpc, :math:`50\\,h`."""
        return jnp.asarray(self._emulator.k_lin[-1]) * self._h

    @property
    def _emulator(self) -> Emulator:
        return load_emulator(self.model_name)

    @property
    def _h(self) -> Float[Array, ""]:
        return _float(self.background.hubble_constant) * C_KM_S / 100

    def _bacco_params(self, z: Real[ArrayLike, ""]) -> dict[str, Float[Array, ""]]:
        """Parameters keyed by ``baccoemu`` name, at one redshift."""
        parameters = self.parameters(self.background)
        h = self._h
        omega_m = _float(self.background.matter_density_parameter(0.0))
        neutrino_mass = _float(self.neutrino_mass)
        p = {
            "hubble": h,
            "omega_cold": omega_m - neutrino_mass / NU_DENSITY / h**2,
            "omega_baryon": _float(parameters.omega_b),
            "ns": _float(parameters.n_s),
            "neutrino_mass": neutrino_mass,
            "w0": _float(parameters.w0),
            "wa": _float(parameters.wa),
        }
        if self.sigma8_cold is not None:
            amplitude = {"sigma8_cold": _float(self.sigma8_cold)}
        else:
            amplitude = {"A_s": 1e-10 * jnp.exp(_float(parameters.ln10as))}
        p["A_s"], p["sigma8_cold"] = amplitudes(self._emulator, p | amplitude)
        p["expfactor"] = 1 / (1 + jnp.asarray(z, dtype=float))
        return p

    def _log_pk_lin(self, p: dict[str, Float[Array, ""]]) -> Float[Array, " k_lin"]:
        return linear_log_pk(self._emulator, p, cold=self.cold)

    def _evaluate(
        self,
        table: _Table,
        log_k_grid: np.ndarray,
        k_h: Float[Array, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Interpolate a per-redshift table linearly in ln k, elementwise.

        ``table`` runs once per element of ``z`` before broadcasting.
        """
        z = jnp.asarray(z)
        shape = jnp.broadcast_shapes(k_h.shape, z.shape)
        rows = jax.vmap(table)(z.ravel())
        index = jnp.broadcast_to(jnp.arange(z.size).reshape(z.shape), shape)
        log_k = jnp.broadcast_to(jnp.log(k_h), shape)
        xp = jnp.asarray(log_k_grid)
        out = jax.vmap(lambda x, i: interp_extrap(x, xp, rows[i]))(
            log_k.ravel(), index.ravel()
        )
        return out.reshape(shape)

    def _k_h(self, k: Real[ArrayLike, "*shape"] | None) -> Float[Array, "*shape"]:
        """Wavenumber in h/Mpc; the largest scale of the linear grid for ``None``."""
        if k is None:
            return jnp.asarray(self._emulator.k_lin[0])
        return jnp.asarray(k) / self._h

    def growth_function(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Linear growth function, :math:`\\sqrt{P_\\mathrm{lin}(k, z) / P_\\mathrm{lin}(k, 0)}`.

        Scale dependent with massive neutrinos.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.
        k : ArrayLike, optional
            Wavenumber in 1/Mpc. ``None`` uses the largest scale of the linear
            emulator, :math:`10^{-4}\\,h\\,\\mathrm{Mpc}^{-1}`.

        Returns
        -------
        Array
            Dimensionless, with the broadcast shape of ``k`` and ``z``.
        """
        if k is not None and self.extrapolate:
            z = jnp.asarray(z, dtype=float)
            return jnp.exp(
                0.5
                * (self._log_pk_extrapolated(k, z) - self._log_pk_extrapolated(k, 0.0))
            )
        log_pk_today = self._log_pk_lin(self._bacco_params(0.0))

        def half_log_ratio(zi: Float[Array, ""]) -> Float[Array, " k_lin"]:
            return 0.5 * (self._log_pk_lin(self._bacco_params(zi)) - log_pk_today)

        log_k_lin = np.log(self._emulator.k_lin)
        return jnp.exp(self._evaluate(half_log_ratio, log_k_lin, self._k_h(k), z))

    def growth_rate(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Linear growth rate :math:`f = \\mathrm{d}\\ln D / \\mathrm{d}\\ln a`.

        A least-squares slope of :math:`\\ln D` over
        :math:`|\\Delta \\ln a| \\le 0.06` around each redshift, which smooths
        over the piecewise-linear networks; see the class notes.

        Parameters
        ----------
        z : ArrayLike
            Redshift, non-negative.
        k : ArrayLike, optional
            Wavenumber in 1/Mpc. ``None`` uses the largest scale of the linear
            emulator, :math:`10^{-4}\\,h\\,\\mathrm{Mpc}^{-1}`.

        Returns
        -------
        Array
            Dimensionless, with the broadcast shape of ``k`` and ``z``.
        """

        def rate(zi: Float[Array, ""]) -> Float[Array, " k_lin"]:
            p = self._bacco_params(zi)
            terms = [
                w * self._log_pk_lin({**p, "expfactor": p["expfactor"] * np.exp(dx)})
                for dx, w in zip(_GROWTH_STENCIL, _GROWTH_WEIGHTS, strict=True)
            ]
            return 0.5 * sum(terms)

        log_k_lin = np.log(self._emulator.k_lin)
        return self._evaluate(rate, log_k_lin, self._k_h(k), z)

    def matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Matter power spectrum :math:`P(k, z)`, total or cold per ``cold``.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive.
        z : ArrayLike
            Redshift, non-negative. Broadcast elementwise against ``k``.

        Returns
        -------
        Array
            :math:`P(k, z)` in :math:`\\mathrm{Mpc}^3`, with the broadcast
            shape of ``k`` and ``z``.
        """
        pk = self.linear_matter_power_spectrum(k, z)
        if not self.nonlinear:
            return pk
        emu = self._emulator
        k_h = self._k_h(k)

        def boost(zi: Float[Array, ""]) -> Float[Array, " k_nl"]:
            p = self._bacco_params(zi)
            log_pk_cold = linear_log_pk(emu, p, cold=True)
            log_pk = log_pk_cold if self.cold else linear_log_pk(emu, p, cold=False)
            return nonlinear_boost(emu, p, log_pk_cold, log_pk, cold=self.cold)

        q = self._evaluate(boost, np.log(emu.k_nl), k_h, z)
        q = jnp.where(k_h < emu.k_nl[0], 1.0, q)
        return q * pk

    def linear_matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
    ) -> Float[Array, "*shape"]:
        """Linear matter power spectrum, total or cold per ``cold``.

        The same whatever ``nonlinear`` is.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive.
        z : ArrayLike
            Redshift, non-negative. Broadcast elementwise against ``k``.

        Returns
        -------
        Array
            :math:`P_\\mathrm{lin}(k, z)` in :math:`\\mathrm{Mpc}^3`, with the
            broadcast shape of ``k`` and ``z``.
        """
        return jnp.exp(self._log_pk_extrapolated(k, z))

    def _log_pk_at(self, k: Array, z: Array) -> Array:
        log_pk_h = self._evaluate(
            lambda zi: self._log_pk_lin(self._bacco_params(zi)),
            np.log(self._emulator.k_lin),
            self._k_h(k),
            z,
        )
        return log_pk_h - 3 * jnp.log(self._h)

    def primordial_power_spectrum(
        self,
        k: Real[ArrayLike, "*shape"],
    ) -> Float[Array, "*shape"]:
        """Power-law primordial spectrum :math:`A_s (k / k_p)^{n_s - 1}`.

        Parameters
        ----------
        k : ArrayLike
            Wavenumber in 1/Mpc, positive. The pivot is
            :math:`k_p = 0.05\\,\\mathrm{Mpc}^{-1}`.

        Returns
        -------
        Array
            :math:`\\mathcal{P}_\\mathcal{R}(k)`, dimensionless, with the shape
            of ``k``.
        """
        p = self._bacco_params(0.0)
        return p["A_s"] * (jnp.asarray(k) / K_PIVOT) ** (p["ns"] - 1)

    def check_bounds(
        self,
        k: Real[ArrayLike, "*k_shape"] | None = None,
        z: Real[ArrayLike, "*z_shape"] = 0.0,
    ) -> None:
        """Raise if the cosmology, ``k`` or ``z`` lies outside the training ranges.

        Performs the validation ``baccoemu`` does on every call. It needs
        concrete values, so call it outside traced code. Checks the nonlinear
        emulator's ranges only when the instance has ``nonlinear=True``.

        Parameters
        ----------
        k : ArrayLike, optional
            Wavenumbers in 1/Mpc that will be evaluated.
        z : ArrayLike, default 0.0
            Redshifts that will be evaluated.

        Raises
        ------
        ValueError
            Listing every parameter or range that is out of bounds.
        """
        emu = self._emulator
        z = np.atleast_1d(np.asarray(z, dtype=float))
        p = {key: float(val) for key, val in self._bacco_params(0.0).items()}
        checks = [(LINEAR_KEYS, emu.bounds_linear, "linear")]
        checks.append((SIGMA8_KEYS, emu.bounds_sigma8, "sigma8"))
        if self.nonlinear:
            checks.append((NONLINEAR_KEYS, emu.bounds_nonlinear, "nonlinear"))

        errors = []
        for keys, bounds, name in checks:
            for key, (lo, hi) in zip(keys, bounds, strict=True):
                vals = 1 / (1 + z) if key == "expfactor" else np.array([p[key]])
                if np.any(vals < lo) or np.any(vals > hi):
                    errors.append(
                        f"{name} emulator: {key} in [{vals.min():.6g}, "
                        f"{vals.max():.6g}] outside [{lo:.6g}, {hi:.6g}]"
                    )
        if k is not None:
            k_h = np.asarray(k, dtype=float) / p["hubble"]
            k_max = emu.k_nl[-1] if self.nonlinear else emu.k_lin[-1]
            if np.min(k_h) < emu.k_lin[0] or np.max(k_h) > k_max:
                errors.append(
                    f"k/h in [{np.min(k_h):.6g}, {np.max(k_h):.6g}] h/Mpc outside "
                    f"[{emu.k_lin[0]:.6g}, {k_max:.6g}]"
                )
        if errors:
            raise ValueError("; ".join(errors))
