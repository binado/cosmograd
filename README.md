# cosmograd

Differentiable cosmology in JAX behind a small protocol contract.

Consumers depend on two protocols, `Background` (expansion history and
distances) and `Perturbations` (growth and power spectra), not on any one
implementation. The core depends on JAX and jaxtyping; emulator backends sit behind
optional extras.

## Install

```console
uv add cosmograd                       # core: protocols + native backgrounds
uv add 'cosmograd[cosmopower]'         # CosmoPower-JAX backend (planned)
uv add 'cosmograd[bacco]'              # bacco emulator backend
```

## Usage

```python
import jax
import jax.numpy as jnp
from cosmograd import FlatLCDM

cosmo = FlatLCDM(h=0.6736, omega_m=0.3153)
z = jnp.linspace(0.1, 2.0, 20)

cosmo.luminosity_distance(z)  # Mpc
jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)  # FlatLCDM-shaped Jacobian
```

With the `bacco` extra, `BaccoEmu` emulates linear and nonlinear matter power
spectra (weights are downloaded on first use):

```python
from cosmograd.perturbations.baccoemu import BaccoEmu

pert = BaccoEmu(h=0.67, omega_cold=0.315, omega_baryon=0.05, ns=0.96, sigma8_cold=0.8)
k = jnp.geomspace(1e-3, 3.0, 100)  # 1/Mpc
z = jnp.linspace(0.0, 1.5, 16)
pert.check_bounds(k, z)  # eager check of the training ranges
pert.matter_power_spectrum(k, z[:, None], nonlinear=True)  # Mpc^3, shape (16, 100)
```

Units: distances in Mpc, wavenumbers in 1/Mpc, H in 1/Mpc (c = 1), power spectra
in Mpc^3. Inputs are not validated; each method documents the range it expects.

To check your own implementation against the contract:

```python
from cosmograd.testing import check_background

check_background(MyBackground(...), z=jnp.linspace(0.0, 3.0, 50))
```

## Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Just](https://just.systems/man/en/packages.html), then run:

```console
just setup
uv run prek install
```

The underlying command is `uv sync --all-groups` if Just is unavailable.

## Development

```console
just fmt        # uv run ruff check --fix . && uv run ruff format .
just lint       # uv run ruff check .
just typecheck  # uv run ty check
just test       # uv run pytest
just hooks      # uv run prek run --all-files
just check      # full validation, tests, and package build
```

Tests check shape and dtype annotations through jaxtyping’s pytest import hook
and beartype. Normal library imports do not enable runtime checking or require
beartype. Shape annotations document conventions; static type checking does not
verify array shapes.
