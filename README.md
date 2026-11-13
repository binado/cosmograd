# cosmograd

Differentiable cosmology in JAX behind a small protocol contract.

Consumers depend on two protocols, `Background` (expansion history and
distances) and `Perturbations` (growth and power spectra), not on any one
implementation. The core depends on JAX, jaxtyping and Diffrax; emulator
backends sit behind optional extras.

## Install

```console
uv add cosmograd                       # core: protocols + native backgrounds and linear power
uv add 'cosmograd[cosmopower]'         # CosmoPower-JAX backend (planned)
uv add 'cosmograd[bacco]'              # bacco emulator backend
```

## Usage

```python
import jax
import jax.numpy as jnp
from cosmograd import FlatLCDM

cosmo = FlatLCDM(H0=67.36, omega_m=0.3153)
z = jnp.linspace(0.1, 2.0, 20)

cosmo.luminosity_distance(z)  # Mpc
jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)  # FlatLCDM-shaped Jacobian
```

The native `EisensteinHu` backend provides linear matter power with EH98 baryon
acoustic oscillations, normalised by the primordial amplitude:

```python
from cosmograd import EisensteinHu

pert = EisensteinHu(cosmo, omega_b=0.0493, n_s=0.9649, ln10as=3.044)
k = jnp.geomspace(1e-4, 1.0, 100)  # 1/Mpc
pert.background is cosmo  # True
pert.matter_power_spectrum(k, z[:, None])  # Mpc^3, shape (20, 100)
pert.growth_function(z)  # normalised to 1 at z = 0
jax.grad(lambda p: p.matter_power_spectrum(0.1, 0.5))(
    pert
)  # nested parameter gradients
```

Here `omega_b` is Ω_b today, with `0 < omega_b < cosmo.omega_m`, and `ln10as`
is ln(10¹⁰ A_s) at the pivot 0.05/Mpc. Growth is scale independent and neglects
radiation, using a Diffrax ODE initialized with D = a at a = 0.001 and checked
through z = 10; background distances retain radiation. Massive neutrinos are
unsupported, and constructing with `nonlinear=True` raises
`NotImplementedError`. For protocol checks on this backend, use
`check_perturbations(pert, k, 0.5, linear_cosmo=pert)`.

The reusable growth solver accepts any differentiable `Background`:

```python
from cosmograd.perturbations import linear_growth

# Matter-era initial conditions require negligible radiation at a_initial.
growth_background = FlatLCDM(H0=67.36, omega_m=0.3153, t_cmb=0.0)
D, f = linear_growth(growth_background, z)  # D(0) = 1, f = d ln D / d ln a
D_md, f = linear_growth(growth_background, z, normalize=False)  # initial D = a
```

The solver uses the supplied H(z) and Ω_m(z), supporting curvature and smooth
dark energy within GR and scale-independent, subhorizon growth. Set
`a_initial` inside matter domination and check sensitivity to this choice.
The default of 0.001 (z = 999) is appropriate only for radiation-free
backgrounds (`t_cmb=0`); with radiation, D and f shift by roughly 1e-4 to 1e-3
at the default, which `rtol` does not control. Such backgrounds need a larger
`a_initial`, late enough for radiation to be negligible but early enough that
dark energy and curvature still are. Redshifts must lie between 0 and
`1/a_initial - 1`, outside which the result is NaN, and may have any shape or
order. Solver failures, such as exceeding `max_steps`, raise an error. Both
forward- and
reverse-mode parameter derivatives are supported. EH keeps its `FlatLCDM`
transfer-function inputs and supplies a radiation-free expansion to this solver.

With the `bacco` extra, `BaccoEmu` emulates linear and nonlinear matter power
spectra (weights are downloaded on first use). Select the spectrum with the
constructor's `nonlinear` option, which defaults to `False` for every backend:

```python
from cosmograd.perturbations.baccoemu import BaccoEmu

pert = BaccoEmu(
    h=0.67,
    omega_cold=0.315,
    omega_baryon=0.05,
    ns=0.96,
    sigma8_cold=0.8,
    nonlinear=True,
)
k = jnp.geomspace(1e-3, 3.0, 100)  # 1/Mpc
z = jnp.linspace(0.0, 1.5, 16)
pert.check_bounds(k, z)  # eager check of the training ranges
pert.matter_power_spectrum(k, z[:, None])  # Mpc^3, shape (16, 100)
```

Units: distances in Mpc, wavenumbers in 1/Mpc, H in 1/Mpc (c = 1), power spectra
in Mpc^3. Inputs are not validated; each method documents the range it expects.

To check your own implementation against the contract:

```python
from cosmograd.testing import check_background

check_background(MyBackground(...), z=jnp.linspace(0.0, 3.0, 50))
```

## Calculation references

The `EisensteinHu` calculations follow:

- [Eisenstein & Hu (1998), *Baryonic Features in the Matter Transfer Function*](https://arxiv.org/abs/astro-ph/9709112),
  equations (2)–(24): equality and drag scales, CDM and baryon contributions,
  acoustic oscillations and Silk damping.
- [Heath (1977)](https://doi.org/10.1093/mnras/179.3.351) and
  [Hamilton (2001), *Formulae for growth factors*](https://arxiv.org/abs/astro-ph/0006089),
  equations (1)–(2): the growing-mode integral for matter plus Λ, used as an
  independent reference in the ODE tests. The Diffrax solver integrates the
  equivalent Sturm–Liouville form (a² E D′)′ = (3/2) a² E Ω_m D in ln a, using
  the variables described in the `linear_growth` notes, so H is never
  differentiated. Matter-era growth starts with D_md = a and D_md′ = a; the
  public growth function is D(z) = D_md(z) / D_md(0).
- [Smith & Angulo (2019), *Precision modelling of the matter power spectrum in a Planck-like Universe*](https://arxiv.org/abs/1807.00040),
  equations (8) and (10): the primordial spectrum (with zero running) and its
  conversion to matter power. Their a g corresponds to D_md here; using
  today-normalised D instead would change the absolute power normalisation.
- [Planck Collaboration (2020), *Planck 2018 results. VI. Cosmological parameters*](https://arxiv.org/abs/1807.06209),
  section 2.1: the pivot k = 0.05/Mpc and ln(10¹⁰ A_s) parameter convention.

Equation references also appear beside the corresponding calculations in
`src/cosmograd/perturbations/einstein_hu.py`.

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
