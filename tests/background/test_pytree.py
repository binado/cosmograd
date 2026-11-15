"""A background is a pytree whose leaves are exactly its free parameters."""

from dataclasses import fields
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from cosmograd import LCDM, FlatLCDM, FlatW0WaCDM, W0WaCDM

from ..cosmologies import COSMOLOGIES, Cosmology, cosmology_params

PERTURBATION_PARAMETERS = ("omega_b", "n_s", "ln10as")
FIELDS = [
    pytest.param(cosmo, field.name, id=f"{name}-{field.name}")
    for name, cosmo in COSMOLOGIES.items()
    for field in fields(cosmo)
]


@pytest.mark.parametrize(
    ("cosmo", "free"),
    [
        (FlatLCDM(H0=67.0, omega_m=0.31), ("H0", "omega_m")),
        (LCDM(H0=67.0, omega_m=0.31, omega_k=0.05), ("H0", "omega_m", "omega_k")),
        (
            FlatW0WaCDM(H0=67.0, omega_m=0.31, w0=-0.9, wa=0.3),
            ("H0", "omega_m", "w0", "wa"),
        ),
        (
            W0WaCDM(H0=67.0, omega_m=0.31, omega_k=0.05, w0=-0.9, wa=0.3),
            ("H0", "omega_m", "omega_k", "w0", "wa"),
        ),
    ],
    ids=["flat-lcdm", "lcdm", "flat-w0wacdm", "w0wacdm"],
)
def test_leaves_are_the_free_parameters(cosmo: Any, free: tuple[str, ...]) -> None:
    expected = [getattr(cosmo, name) for name in (*free, *PERTURBATION_PARAMETERS)]
    assert jax.tree.leaves(cosmo) == expected


@pytest.mark.parametrize(
    ("cosmo", "fixed"),
    [
        (FlatLCDM(H0=67.0, omega_m=0.31), {"omega_k": 0.0, "w0": -1.0, "wa": 0.0}),
        (FlatW0WaCDM(H0=67.0, omega_m=0.31), {"omega_k": 0.0}),
        (LCDM(H0=67.0, omega_m=0.31), {"w0": -1.0, "wa": 0.0}),
    ],
    ids=["flat-lcdm", "flat-w0wacdm", "lcdm"],
)
def test_constrained_class_fixes_the_other_parameters(
    cosmo: Any, fixed: dict[str, float]
) -> None:
    assert {name: getattr(cosmo, name) for name in fixed} == fixed


@pytest.mark.parametrize("cosmo", cosmology_params())
def test_leaves_are_floating_point(cosmo: Cosmology) -> None:
    for leaf in jax.tree.leaves(cosmo):
        assert jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)


@pytest.mark.parametrize(("cosmo", "name"), FIELDS)
def test_with_parameters_replaces_only_the_given_field(
    cosmo: Cosmology, name: str
) -> None:
    value = getattr(cosmo, name) + 0.05
    new = cosmo.with_parameters(**{name: value})

    assert type(new) is type(cosmo)
    assert getattr(new, name) == value
    for field in fields(cosmo):
        if field.name != name:
            assert getattr(new, field.name) == getattr(cosmo, field.name)
    assert getattr(cosmo, name) != value  # original is untouched
