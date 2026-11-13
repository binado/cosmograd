"""``with_parameters`` on every parametric background."""

import inspect
from dataclasses import fields

import pytest

from cosmograd import LCDM, FlatLCDM, FlatW0WaCDM, W0WaCDM
from cosmograd.background import Background

PERTURBATION_PARAMETERS = {"omega_b": 0.049, "n_s": 0.965, "ln10as": 3.044}
CASES = [
    FlatLCDM(H0=67.0, omega_m=0.31, **PERTURBATION_PARAMETERS),
    LCDM(H0=67.0, omega_m=0.31, omega_k=0.02, **PERTURBATION_PARAMETERS),
    FlatW0WaCDM(H0=67.0, omega_m=0.31, w0=-0.9, wa=0.2, **PERTURBATION_PARAMETERS),
    W0WaCDM(
        H0=67.0, omega_m=0.31, omega_k=0.02, w0=-0.9, wa=0.2, **PERTURBATION_PARAMETERS
    ),
]
FIELD_CASES = [(c, f.name) for c in CASES for f in fields(c)]


@pytest.fixture(params=CASES, ids=lambda c: type(c).__name__)
def cosmo(request: pytest.FixtureRequest) -> Background:
    return request.param


def test_no_arguments_returns_equal_copy(cosmo) -> None:
    new = cosmo.with_parameters()
    assert new == cosmo
    assert new is not cosmo


def test_signature_matches_dataclass_fields(cosmo) -> None:
    params = inspect.signature(cosmo.with_parameters).parameters
    assert set(params) == {f.name for f in fields(cosmo)}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())


@pytest.mark.parametrize(
    ("cosmo", "name"),
    FIELD_CASES,
    ids=lambda x: x if isinstance(x, str) else type(x).__name__,
)
def test_replaces_only_the_given_field(cosmo, name: str) -> None:
    value = getattr(cosmo, name) + 0.05
    new = cosmo.with_parameters(**{name: value})

    assert type(new) is type(cosmo)
    assert getattr(new, name) == value
    for f in fields(cosmo):
        if f.name != name:
            assert getattr(new, f.name) == getattr(cosmo, f.name)
    assert getattr(cosmo, name) != value  # original is untouched
