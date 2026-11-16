# cosmograd

Differentiable cosmology in JAX, with interchangeable background and perturbation models.

## Installation

```console
uv add cosmograd
uv add 'cosmograd[cosmopower]'  # CosmoPower-JAX backend
uv add 'cosmograd[bacco]'       # bacco emulator backend
```

## Usage

```python
import jax
from cosmograd import EisensteinHu, FlatLCDM

cosmo = FlatLCDM(H0=67.36, omega_m=0.3153)
cosmo.luminosity_distance(1.0)  # Mpc
jax.grad(lambda c: c.comoving_distance(1.0))(cosmo)
EisensteinHu(cosmo).matter_power_spectrum(0.1, 0.5)  # Mpc^3
```

Or wrap your own parameter pytree and Hubble rate with `GenericBackground`:

```python
import jax
import jax.numpy as jnp
from cosmograd import GenericBackground

def hubble(p, z):
    e2 = p["omega_m"] * (1 + z) ** 3 + 1 - p["omega_m"]
    return p["h"] / 2997.92458 * jnp.sqrt(e2)

bg = GenericBackground({"h": 0.6736, "omega_m": 0.3153}, hubble, lambda p: p["omega_m"])
jax.grad(lambda b: b.luminosity_distance(1.0))(bg).params
```

See the API docstrings for parameter conventions, units and valid ranges.

## Features

- Background expansion and distances for curved and flat cosmologies.
- Linear growth and Eisenstein–Hu matter power spectra.
- Optional CosmoPower and bacco emulator backends.
- JAX differentiation through cosmological parameters.

## Caveats

Perturbation backends assume a flat universe. Set the cosmological parameters
explicitly when computing spectra; otherwise their defaults are used. The growth
solver's initial conditions require matter domination; see its docstring for
radiation and redshift constraints. Emulator weights may download on first use.

## References

- CosmoPower-JAX: [Piras & Spurio Mancini (2023)](https://arxiv.org/abs/2305.06347), [CosmoPower paper](https://arxiv.org/abs/2106.03846), and [repository](https://github.com/dpiras/cosmopower-jax).
- baccoemu: [linear power emulator](https://arxiv.org/abs/2104.14568), [nonlinear power emulator](https://arxiv.org/abs/2004.06245), and [repository](https://bitbucket.org/rangulo/baccoemu/).
- [Eisenstein & Hu (1998), Baryonic Features in the Matter Transfer Function](https://arxiv.org/abs/astro-ph/9709112)
- [Smith & Angulo (2019), Precision modelling of the matter power spectrum](https://arxiv.org/abs/1807.00040)
- [Planck Collaboration (2020), Planck 2018 cosmological parameters](https://arxiv.org/abs/1807.06209)

## License

[MIT](LICENSE)
