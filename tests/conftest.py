"""Test configuration. The package itself never changes JAX precision."""

import jax
import pytest

from cosmograd.perturbations.baccoemu._emulator import load_emulator

from .optional import installed

jax.config.update("jax_enable_x64", True)


def pytest_configure(config: pytest.Config) -> None:
    """Fetch every baccoemu emulator once, before any test or xdist worker runs.

    The emulators cosmograd uses go through its own cache, which is safe under
    concurrency. The baryonic ones are only needed by cloelib, which loads them
    through ``baccoemu``'s downloader. That downloads into the install
    directory, racing between workers, and disables certificate checks by
    replacing ``ssl``'s default context, which ``urllib`` ignores once any other
    download, such as cloelib's CosmoPower emulators, has built its opener. So
    the downloads must come first. The variants are those cloelib loads.
    """
    if hasattr(config, "workerinput") or not installed("baccoemu"):
        return
    from baccoemu.matter_powerspectrum import Matter_powerspectrum

    for model in ("Angulo2021", "Arico2023"):
        load_emulator(model)
    Matter_powerspectrum(baryonic_boost=False, verbose=False)
    for nonlinear in ("Angulo2021", "Arico2023"):
        for baryonic in ("Arico2021", "Burger2025"):
            Matter_powerspectrum(
                nonlinear_model_name=nonlinear,
                baryonic_model_name=baryonic,
                baryonic_boost=True,
                verbose=False,
            )
