"""Cosmologies shared by the test suite.

Every entry sets all the parameters a perturbation backend needs, so each one is
usable as input to both backgrounds and perturbations. Reference comparisons
only need a fixed, plausible set of inputs, not exact published values.
"""

from typing import Literal

import pytest
from _pytest.mark.structures import ParameterSet

from cosmograd import LCDM, FlatLCDM, FlatW0WaCDM, W0WaCDM
from cosmograd import cosmologies as standard

Cosmology = FlatLCDM | LCDM | FlatW0WaCDM | W0WaCDM

H0, OMEGA_M = 67.66, 0.3111

COSMOLOGIES: dict[str, Cosmology] = {
    "planck18": standard.Planck18,
    "planck15": standard.Planck15,
    "planck13": standard.Planck13,
    "planck18-w0wa": FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=-0.9, wa=0.2),
    # w crosses -1 at a = 0.8, which CAMB only handles with its PPF fluid.
    "planck18-phantom-crossing": FlatW0WaCDM(H0=H0, omega_m=OMEGA_M, w0=-1.1, wa=0.5),
    "planck18-open": LCDM(H0=H0, omega_m=OMEGA_M, omega_k=0.05),
    "planck18-open-w0wa": W0WaCDM(
        H0=H0, omega_m=OMEGA_M, omega_k=0.05, w0=-0.9, wa=0.2
    ),
    "planck18-closed": LCDM(H0=H0, omega_m=OMEGA_M, omega_k=-0.05),
}
"""Reference cosmologies, by name: flat and curved, Λ and evolving dark energy."""

EMULATOR_COSMOLOGY = FlatLCDM(
    H0=67.0, omega_m=0.31, omega_b=0.049, n_s=0.965, ln10as=3.044
)
"""A cosmology inside every perturbation backend's supported range."""


def cosmology_params(
    kind: Literal["flat", "flat_lcdm"] | None = None,
) -> list[ParameterSet]:
    """The reference cosmologies as ``pytest.param`` objects with readable ids.

    Parameters
    ----------
    kind : {"flat", "flat_lcdm"}, optional
        ``None`` returns every cosmology, ``"flat"`` the flat ones (ΛCDM and
        w0wa) and ``"flat_lcdm"`` the flat ΛCDM ones.

    Returns
    -------
    list of pytest.param
        One per cosmology, with its name as id.
    """
    classes = {
        None: (FlatLCDM, LCDM, FlatW0WaCDM, W0WaCDM),
        "flat": (FlatLCDM, FlatW0WaCDM),
        "flat_lcdm": (FlatLCDM,),
    }
    if kind not in classes:
        raise ValueError(f"unknown kind {kind!r}")
    return [
        pytest.param(cosmo, id=name)
        for name, cosmo in COSMOLOGIES.items()
        if isinstance(cosmo, classes[kind])
    ]
