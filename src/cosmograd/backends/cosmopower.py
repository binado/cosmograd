"""CosmoPower-JAX power-spectrum adapter. Needs the ``cosmopower`` extra.

Not implemented yet. The adapter will map parameters, convert h/Mpc to 1/Mpc
and pair the emulated spectrum with a native :class:`~cosmograd.Background`.
"""

from __future__ import annotations

from types import ModuleType

from ._optional import require

__all__ = ["load_cosmopower"]

EXTRA: str = "cosmopower"


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
    return require("cosmopower_jax", EXTRA)
