"""Regenerate frozen references using a checkout of corrtracers.

Run with corrtracers installed at revision
661c367c64b3f329dc431d303f9a62f2e80b0868. Not needed to run tests.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from corrtracers.cosmology.growth import Growth  # ty: ignore[import-not-found]
from corrtracers.cosmology.pk.eisenstein_hu import (  # ty: ignore[import-not-found]
    EisensteinHu,
    eh98_transfer,
)

jax.config.update("jax_enable_x64", True)

k = jnp.array([1e-5, 1e-3, 0.01, 0.03, 0.07, 0.1, 0.2, 0.5, 1.0, 10.0])
z = jnp.array([0.0, 0.5, 1.0, 3.0, 10.0])
cases = []
for h, om, ob, ns, ln10as, temperature in [
    (0.6736, 0.3153, 0.0493, 0.9649, 3.044, 2.7255),
    (0.72, 0.25, 0.04, 1.0, 3.1, 2.7255),
    (0.62, 0.4, 0.07, 0.94, 2.9, 2.8),
]:
    cosmo = {
        "h": h,
        "omega_b": ob * h**2,
        "omega_cdm": (om - ob) * h**2,
        "n_s": ns,
        "ln10As": ln10as,
    }
    growth = Growth.build(jnp.linspace(np.log(1e-4), 0.0, 512), om)
    cases.append(
        {
            "parameters": {
                "H0": 100 * h,
                "omega_m": om,
                "omega_b": ob,
                "n_s": ns,
                "ln10as": ln10as,
                "t_cmb": temperature,
            },
            "transfer": np.asarray(
                eh98_transfer(k, ob * h**2, om * h**2, temperature)
            ).tolist(),
            "power_z0": np.asarray(EisensteinHu(t_cmb=temperature)(cosmo, k)).tolist(),
            "growth": np.asarray(growth.D(z)).tolist(),
            "rate": np.asarray(growth.f(z)).tolist(),
        }
    )

Path(__file__).with_name("eisenstein_hu_reference.json").write_text(
    json.dumps(
        {
            "revision": "661c367c64b3f329dc431d303f9a62f2e80b0868",
            "k": np.asarray(k).tolist(),
            "z": np.asarray(z).tolist(),
            "cases": cases,
        },
        indent=2,
    )
    + "\n"
)
