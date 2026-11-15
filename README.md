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

- [Eisenstein & Hu (1998), Baryonic Features in the Matter Transfer Function](https://arxiv.org/abs/astro-ph/9709112)
- [Smith & Angulo (2019), Precision modelling of the matter power spectrum](https://arxiv.org/abs/1807.00040)
- [Planck Collaboration (2020), Planck 2018 cosmological parameters](https://arxiv.org/abs/1807.06209)

## License

[MIT](LICENSE)
