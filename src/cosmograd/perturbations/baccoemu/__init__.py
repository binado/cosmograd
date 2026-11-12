"""JAX port of the bacco emulators. Needs the ``bacco`` extra.

Not implemented yet: the network evaluation moves here from baccoemu-jax.
``baccoemu`` itself is used only to download and load the weights.
"""

from __future__ import annotations

from types import ModuleType

from .._optional import require

__all__ = ["load_baccoemu"]

EXTRA: str = "bacco"


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
    return require("baccoemu", EXTRA)
