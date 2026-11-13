"""Loading the bacco network weights.

``baccoemu`` is used only here, once per model, to download and read the
trained weights. Everything downstream is pure ``jax.numpy``. Weights are kept
as NumPy arrays so they enter traced code as constants, never as pytree leaves.
"""

import importlib
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Literal, NamedTuple, Protocol, runtime_checkable

import jax
import numpy as np
from jax import Array
from jaxtyping import Float

from ._cache import NONLINEAR_EMULATORS, fetch_emulators

__all__ = [
    "LINEAR_KEYS",
    "MLP",
    "NONLINEAR_KEYS",
    "SIGMA8_KEYS",
    "Emulator",
    "ModelName",
    "load_baccoemu",
    "load_emulator",
    "mlp_apply",
]

ModelName = Literal["Angulo2021", "Arico2023"]
"""Nonlinear emulators shipped with ``baccoemu``."""

MLP = tuple[tuple[np.ndarray, np.ndarray], ...]
"""Dense ReLU network as ``((W_0, b_0), ..., (W_n, b_n))``; last layer is linear."""

LINEAR_KEYS = (
    "omega_cold",
    "omega_baryon",
    "hubble",
    "neutrino_mass",
    "w0",
    "wa",
    "expfactor",
)
SIGMA8_KEYS = (
    "omega_cold",
    "omega_baryon",
    "ns",
    "hubble",
    "neutrino_mass",
    "w0",
    "wa",
)
NONLINEAR_KEYS = (
    "omega_cold",
    "sigma8_cold",
    "omega_baryon",
    "ns",
    "hubble",
    "neutrino_mass",
    "w0",
    "wa",
    "expfactor",
)


class Emulator(NamedTuple):
    """Weights, k grids and parameter bounds of the bacco emulators.

    Wavenumbers are in h/Mpc and spectra in (Mpc/h)^3, as in ``baccoemu``.

    Attributes
    ----------
    sigma8 : MLP
        Maps ``SIGMA8_KEYS`` to (sigma8_cold, sigma8_tot) at ``A_s = 1e-9``.
    linear_cold, linear_total : MLP
        Linear cold-matter and total-matter P(k) networks on ``k_lin``, at
        ``A_s = 1e-9`` and ``ns = 1``, in log space without their mean.
    linear_cold_mean, linear_total_mean : np.ndarray
        Output means added back to the linear networks.
    no_wiggles : MLP
        Network for log(P_nowiggle / P_lin) on ``k_lin``.
    no_wiggles_mean : np.ndarray
        Output mean added back to the no-wiggles network.
    nonlinear : MLP
        Network for the log of the nonlinear-to-smeared-BAO ratio on ``k_nl``.
        Its inputs are rescaled to the unit hypercube by ``bounds_nonlinear``.
    nonlinear_mean : np.ndarray
        Output mean added back to the nonlinear network.
    k_lin, k_nl : np.ndarray
        k grids [h/Mpc] of the linear and nonlinear networks.
    bounds_linear, bounds_sigma8, bounds_nonlinear : np.ndarray
        ``(n_keys, 2)`` parameter bounds, ordered as the matching ``*_KEYS``.
    """

    sigma8: MLP
    linear_cold: MLP
    linear_cold_mean: np.ndarray
    linear_total: MLP
    linear_total_mean: np.ndarray
    no_wiggles: MLP
    no_wiggles_mean: np.ndarray
    nonlinear: MLP
    nonlinear_mean: np.ndarray
    k_lin: np.ndarray
    k_nl: np.ndarray
    bounds_linear: np.ndarray
    bounds_sigma8: np.ndarray
    bounds_nonlinear: np.ndarray


@runtime_checkable
class _KerasLikeModel(Protocol):
    """``baccoemu.utils.load_model``: holds a list of ``(kernel, bias)`` pairs."""

    params: list[tuple[np.ndarray, np.ndarray]]


def _mlp(model: _KerasLikeModel) -> MLP:
    return tuple((np.asarray(w), np.asarray(b)) for w, b in model.params)


def load_baccoemu() -> ModuleType:
    """Import ``baccoemu`` on first use.

    Returns
    -------
    ModuleType
        The ``baccoemu`` module.

    Raises
    ------
    ImportError
        If the ``bacco`` extra is not installed.
    """
    try:
        # Import a real submodule: weights downloaded into an uninstalled
        # ``baccoemu`` folder would otherwise import as a namespace package.
        importlib.import_module("baccoemu.matter_powerspectrum")
        return importlib.import_module("baccoemu")
    except ImportError as err:
        raise ImportError(
            "'baccoemu' is required for this backend. "
            "Install it with `pip install 'cosmograd[bacco]'`."
        ) from err


def _check_keys(name: str, got: list[str], expected: tuple[str, ...]) -> None:
    if tuple(got) != expected:
        raise ValueError(
            f"baccoemu {name} emulator expects inputs {got}, but cosmograd "
            f"assumes {list(expected)}. The installed baccoemu is not supported."
        )


@cache
def load_emulator(model_name: ModelName = "Angulo2021") -> Emulator:
    """Load the bacco emulators, downloading the weights on first use.

    The weights are kept in ``$XDG_CACHE_HOME/cosmograd/baccoemu`` (see
    ``_cache``) and symlinked into ``baccoemu``'s install directory. Results
    are cached per ``model_name``.

    Parameters
    ----------
    model_name : {"Angulo2021", "Arico2023"}, default "Angulo2021"
        Nonlinear emulator to load.

    Returns
    -------
    Emulator
        Network weights, k grids and parameter bounds.

    Raises
    ------
    ImportError
        If the ``bacco`` extra is not installed.
    ValueError
        If the installed ``baccoemu`` uses a layout this port does not know.
    """
    mp = load_baccoemu().matter_powerspectrum
    fetch_emulators(
        [
            "cold_matter_linear_emu_1.0.2",
            "total_matter_linear_emu_1.0.2",
            "sigma8_emu_2.0.2",
            "no_wiggles_emu_1.0.2",
            NONLINEAR_EMULATORS[model_name],
        ],
        Path(mp.__file__).parent,
    )
    lin = mp.load_linear_emu(verbose=False)
    s8 = mp.load_sigma8_emu(verbose=False)
    nw = mp.load_no_wiggles_emu(verbose=False)
    nl = mp.load_nonlinear_emu(verbose=False, model_name=model_name)

    _check_keys("linear", lin["keys"], LINEAR_KEYS)
    _check_keys("no-wiggles", nw["keys"], LINEAR_KEYS)
    _check_keys("sigma8", s8["keys"], SIGMA8_KEYS)
    _check_keys("nonlinear", nl["keys"], NONLINEAR_KEYS)
    if not np.array_equal(lin["k"], nw["k"]):
        raise ValueError("Linear and no-wiggles emulators use different k grids.")
    scalers = (lin["scaler_cold"], lin["scaler_tot"], nw["scaler"], nl["scaler"])
    if any(scaler.std is not None for scaler in scalers):
        raise ValueError("Output scalers with a std are not supported.")

    return Emulator(
        sigma8=_mlp(s8["model"]),
        linear_cold=_mlp(lin["model_cold"]),
        linear_cold_mean=np.asarray(lin["scaler_cold"].mean),
        linear_total=_mlp(lin["model_tot"]),
        linear_total_mean=np.asarray(lin["scaler_tot"].mean),
        no_wiggles=_mlp(nw["model"]),
        no_wiggles_mean=np.asarray(nw["scaler"].mean),
        nonlinear=_mlp(nl["model"]),
        nonlinear_mean=np.asarray(nl["scaler"].mean),
        k_lin=np.asarray(lin["k"], dtype=np.float64),
        k_nl=np.asarray(nl["k"], dtype=np.float64),
        bounds_linear=np.asarray(lin["bounds"], dtype=np.float64),
        bounds_sigma8=np.asarray(s8["bounds"], dtype=np.float64),
        bounds_nonlinear=np.asarray(nl["bounds"], dtype=np.float64),
    )


def mlp_apply(mlp: MLP, x: Float[Array, " n_in"]) -> Float[Array, " n_out"]:
    """Evaluate a dense ReLU network with a linear output layer.

    Parameters
    ----------
    mlp : MLP
        Layer weights and biases.
    x : Array
        Input of shape ``(n_in,)``.

    Returns
    -------
    Array
        Output of shape ``(n_out,)``, in the precision of ``x`` (the weights
        are float32 and promote to it).
    """
    for w, b in mlp[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = mlp[-1]
    return x @ w + b
