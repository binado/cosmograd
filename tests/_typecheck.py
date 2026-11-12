"""Runtime checker used by the jaxtyping pytest import hook."""

from beartype import beartype as typecheck

__all__ = ["typecheck"]
