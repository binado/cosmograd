"""Runtime checker used by the jaxtyping pytest import hook."""

from collections.abc import Callable
from typing import Any

from beartype import beartype


def typecheck[F: Callable[..., Any]](function: F) -> F:
    """Check signatures except the dataclass constructor used by JAX unflattening."""
    # JAX reconstructs FlatLCDM pytrees with temporary non-array sentinels when
    # differentiating. Only its generated constructor needs this exception;
    # ordinary methods, properties, and all other functions remain checked.
    if (
        function.__module__ == "cosmograd.background.flat_lambdacdm"
        and getattr(function, "__qualname__", None) == "FlatLCDM.__init__"
    ):
        return function
    return beartype(function)
