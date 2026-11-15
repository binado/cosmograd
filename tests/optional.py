"""Helpers for backends whose dependencies are optional extras."""

import importlib.util

import pytest

__all__ = ["installed", "requires"]


def installed(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # the parent package of a dotted name is missing
        return False


def requires(module: str) -> pytest.MarkDecorator:
    """Skip a test whose optional dependency is not installed."""
    return pytest.mark.skipif(
        not installed(module),
        reason=f"optional dependency {module!r} is not installed",
    )
