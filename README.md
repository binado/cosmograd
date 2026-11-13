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

`FlatW0WaCDM` adds CPL dark energy, $w(a) = w_0 + w_a (1 - a)$, with `w0` and `wa`
as extra differentiable parameters:

```python
from cosmograd import FlatW0WaCDM

cosmo = FlatW0WaCDM(H0=67.36, omega_m=0.3153, w0=-0.9, wa=0.2)
jax.grad(lambda c: c.comoving_distance(1.0))(cosmo).w0
```

`W0WaCDM` is the general background, adding a curvature parameter `omega_k`;
`LCDM`, `FlatW0WaCDM` and `FlatLCDM` are the same model with `omega_k`, `w0`
and `wa` fixed, so they are leaves of the pytree only for the parameters they
leave free. Curvature enters the expansion rate and the transverse distances
(`luminosity_distance`, `angular_diameter_distance`); the inverse of
`comoving_distance` is the line-of-sight one.

```python
from cosmograd import W0WaCDM

cosmo = W0WaCDM(H0=67.36, omega_m=0.3153, omega_k=0.02)
cosmo.luminosity_distance(z)
```

Every class also takes the optional `omega_b`, `n_s` and `ln10as`, so that one
object holds the whole cosmology. No background method reads them, and left
unset they are not part of the pytree. Standard cosmologies carry them:

```python
from cosmograd.cosmologies import Planck18

Planck18.luminosity_distance(z)
Planck18.with_parameters(omega_m=0.30)  # a modified copy
```

Here `omega_b` is $\Omega_b$ today, with `0 < omega_b < omega_m`, and `ln10as` is
$\ln(10^{10} A_s)$ at the pivot $0.05\,\mathrm{Mpc}^{-1}$.

The perturbation backends below take such a cosmology plus their own settings,
and raise `ValueError` when a spectrum needs a parameter that is unset. They
assume a flat universe: `EisensteinHu` and `BaccoEmu` accept either flat class;
`CosmoPower` is trained on $\Lambda\mathrm{CDM}$ and takes a `FlatLCDM` only.

The native `EisensteinHu` backend provides linear matter power with EH98 baryon
acoustic oscillations, normalised by the primordial amplitude:

```python
from cosmograd import EisensteinHu

pert = EisensteinHu(Planck18)
k = jnp.geomspace(1e-4, 1.0, 100)  # 1/Mpc
pert.background is Planck18  # True
pert.matter_power_spectrum(k, z[:, None])  # Mpc^3, shape (20, 100)
pert.growth_function(z)  # normalised to 1 at z = 0
# One FlatLCDM-shaped gradient over H0, omega_m, omega_b, n_s and ln10as.
jax.grad(lambda c: EisensteinHu(c).matter_power_spectrum(0.1, 0.5))(Planck18)
```

Growth is scale independent and neglects
radiation, using a Diffrax ODE initialized with $D = a$ at $a = 0.001$ and checked
through $z = 10$; background distances retain radiation. Massive neutrinos are
unsupported, and constructing with `nonlinear=True` raises
`NotImplementedError`. For protocol checks on this backend, use
`check_perturbations(pert, k, 0.5)`.

The reusable growth solver accepts any differentiable `Background`:

```python
from cosmograd.perturbations import linear_growth

# Matter-era initial conditions require negligible radiation at a_initial.
growth_background = FlatLCDM(H0=67.36, omega_m=0.3153, t_cmb=0.0)
D, f = linear_growth(growth_background, z)  # D(0) = 1, f = d ln D / d ln a
D_md, f = linear_growth(growth_background, z, normalize=False)  # initial D = a
```

The solver uses the supplied $H(z)$ and $\Omega_m(z)$, supporting curvature and smooth
dark energy within GR and scale-independent, subhorizon growth. Set
`a_initial` inside matter domination and check sensitivity to this choice.
The default of 0.001 ($z = 999$) is appropriate only for radiation-free
backgrounds (`t_cmb=0`); with radiation, $D$ and $f$ shift by roughly $10^{-4}$ to
$10^{-3}$ at the default, which `rtol` does not control. Such backgrounds need a larger
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

pert = BaccoEmu(Planck18, nonlinear=True)
k = jnp.geomspace(1e-3, 3.0, 100)  # 1/Mpc
z = jnp.linspace(0.0, 1.5, 16)
pert.check_bounds(k, z)  # eager check of the training ranges
pert.matter_power_spectrum(k, z[:, None])  # Mpc^3, shape (16, 100)
pert.linear_matter_power_spectrum(k, z[:, None])  # linear, whatever `nonlinear` is
```

`BaccoEmu` also takes `neutrino_mass` (the cosmology's `omega_m` is then the
total matter density, $\Omega_{cb} + \Omega_\nu$) and `sigma8_cold`, which sets the amplitude
in place of the cosmology's `ln10as`. With the `cosmopower` extra,
`CosmoPower(Planck18, c_min=3.13, eta_0=0.603, nonlinear=True)` works the same
way, with the HMcode feedback parameters as its own differentiable fields.

Units: distances in Mpc, wavenumbers in $\mathrm{Mpc}^{-1}$, $H$ in $\mathrm{Mpc}^{-1}$
($c = 1$), power spectra in $\mathrm{Mpc}^3$. Inputs are not validated; each method documents the range it expects.

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
  equations (1)–(2): the growing-mode integral for matter plus $\Lambda$, used as an
  independent reference in the ODE tests. The Diffrax solver integrates the
  equivalent Sturm–Liouville form $(a^2 E D')' = \tfrac{3}{2} a^2 E \Omega_m D$ in
  $\ln a$, using the variables described in the `linear_growth` notes, so $H$ is
  never differentiated. Matter-era growth starts with $D_\mathrm{md} = a$ and
  $D_\mathrm{md}' = a$; the public growth function is
  $D(z) = D_\mathrm{md}(z) / D_\mathrm{md}(0)$.
- [Smith & Angulo (2019), *Precision modelling of the matter power spectrum in a Planck-like Universe*](https://arxiv.org/abs/1807.00040),
  equations (8) and (10): the primordial spectrum (with zero running) and its
  conversion to matter power. Their $a g$ corresponds to $D_\mathrm{md}$ here; using
  today-normalised $D$ instead would change the absolute power normalisation.
- [Planck Collaboration (2020), *Planck 2018 results. VI. Cosmological parameters*](https://arxiv.org/abs/1807.06209),
  section 2.1: the pivot $k = 0.05\,\mathrm{Mpc}^{-1}$ and $\ln(10^{10} A_s)$ parameter
  convention.

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

### Spectrum comparison notebook

Install the notebook dependencies with `uv sync --group notebook`, then open
[`notebooks/perturbation_benchmark.py`](notebooks/perturbation_benchmark.py) in a
notebook editor that supports Jupytext `py:percent` scripts (for example, VS Code's
Python Interactive window). Select the repository's `.venv` kernel and run all
cells. The first code cell contains imports; the second holds all editable
settings and enables Retina figures. First use may download emulator weights
and requires internet access.
The notebook compares linear and nonlinear spectra against CAMB at a shared
primordial amplitude; it does not benchmark runtime.

Only the percent script is maintained. To execute a temporary notebook from the
repository root:

```console
uv run --group notebook jupytext --to ipynb notebooks/perturbation_benchmark.py -o /tmp/perturbation_benchmark.ipynb
uv run --group notebook jupyter nbconvert --to notebook --execute /tmp/perturbation_benchmark.ipynb --output /tmp/perturbation_benchmark.executed.ipynb --ExecutePreprocessor.timeout=600
```

### Repository checks

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
