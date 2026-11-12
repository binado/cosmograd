"""JAX port of the bacco emulators. Needs the ``bacco`` extra.

Not implemented yet: the network evaluation moves here from baccoemu-jax.
``baccoemu`` itself is used only to download and load the weights.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["load_baccoemu"]


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
        return importlib.import_module("baccoemu")
    except ImportError as err:
        raise ImportError(
            "'baccoemu' is required for this backend. "
            "Install it with `pip install 'cosmograd[bacco]'`."
        ) from err
