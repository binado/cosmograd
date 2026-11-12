"""CosmoPower-JAX power-spectrum adapter. Needs the ``cosmopower`` extra.

Not implemented yet. The adapter will map parameters, convert h/Mpc to 1/Mpc
and pair the emulated spectrum with a native :class:`~cosmograd.Background`.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["load_cosmopower"]


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
