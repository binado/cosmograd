"""Lazy import of optional backend dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["require"]


def require(module: str, extra: str) -> ModuleType:
    """Import an optional dependency, naming the extra that provides it.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"cosmopower_jax"``.
    extra : str
        Name of the ``cosmograd`` extra that installs it, e.g. ``"cosmopower"``.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If ``module`` is not installed. The message says to install
        ``cosmograd[extra]``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"{module!r} is required for this backend. "
            f"Install it with `pip install 'cosmograd[{extra}]'`."
        ) from err
