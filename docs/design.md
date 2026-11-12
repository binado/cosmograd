# JAX cosmology interface: design

Nov 12, 2026 · @Bernardo

## Summary

A separate JAX package defines two protocols, `Background` and `Perturbations`, and ships the implementations I maintain. corrtracers and gravitational-wave codes depend on the protocols, not on any one implementation.

| Topic | Decision |
| --- | --- |
| Packaging | Separate package, JAX-only core, each emulator backend behind an optional extra |
| Object model | One frozen dataclass per cosmology, registered as a pytree; its only leaves are the parameters |
| Contract | Two protocols: `Background` and `Perturbations`. They list behaviour only, no input parameters |
| Parameters | Typed fields on each implementation, with component slots for physics that varies |
| Names | Verbose: `comoving_distance`, `growth_function`, `matter_power_spectrum` |
| Separability | Not part of the contract. `matter_power_spectrum(k, z)` is general |
| Growth | Takes `z`, with an optional keyword `k` for scale dependence |
| Backends | Live in the package under `backends/`, imported lazily, no registry |

The package is named `cosmograd`; this document also calls it "the package".

## Goals and scope

The package gives any JAX code a differentiable cosmology through one small contract, with the emulator dependencies kept optional.

Two consumers drive the design:

- **corrtracers** needs distances, growth and the matter power spectrum to compute angular spectra and Fisher matrices.
- **Gravitational-wave analyses** without clustering need only the background: distances and volume elements.

In scope:

- The two protocols and a conformance test suite for them.
- Native JAX backgrounds and the Eisenstein-Hu spectrum, installed with the core.
- An adapter for CosmoPower-JAX and a JAX implementation of the bacco emulators, each behind an extra.

Out of scope:

- Boltzmann codes (CAMB, CLASS) as backends. Fixed tables from them remain possible through a tabulated implementation.
- Angular spectra, tracers and integrators. Those stay in corrtracers.
- A parameter alias system or a string-keyed engine registry.

## Object model

A cosmology is one frozen dataclass, registered as a pytree, whose only leaves are its parameters. Static choices are meta fields, and every method is a pure function of the leaves.

```python
@partial(
    register_dataclass, data_fields=["h", "omega_m"], meta_fields=["t_cmb", "n_eff"]
)
@dataclass(frozen=True)
class FlatLCDM(Background):
    h: ArrayLike
    omega_m: ArrayLike
    t_cmb: float = 2.7255
    n_eff: float = 3.044

    def hubble_parameter(self, z): ...
    def comoving_distance(self, z): ...


cosmo = FlatLCDM(h=0.67, omega_m=0.31)
jax.jacfwd(lambda c: c.luminosity_distance(z))(cosmo)  # FlatLCDM-shaped Jacobian
```

This replaces the `Cosmology` and `CosmoState` pair that corrtracers has today. There is no separate state type and no `grids` argument.

**Where the per-parameter work goes.** Methods recompute from the parameters on each call, using a small internal grid fixed by meta fields. A consumer that knows its own grids evaluates once per parameter point and keeps the arrays as locals:

```python
# inside corrtracers' Model.cl(theta)
chi = cosmo.background.comoving_distance(grids.z)
pk = cosmo.matter_power_spectrum(grids.k, grids.z)
```

**Rejected alternative: derived tables as leaves.** Storing the distance or power tables as pytree leaves, built in the constructor, also gives one object. It was rejected for two reasons:

- `jax.grad(f)(cosmo)` would treat the tables as independent of the parameters and return the wrong derivative.
- `dataclasses.replace` or `tree_map` on a parameter would leave the tables stale.

**Costs accepted.**

- Two separate eager calls each rebuild the background table. This is a few hundred points; inside one jitted function the consumer hoists it.
- Every field is a leaf, so fixed parameters still get Jacobian columns unless selected out. Differentiating a subset is `jacfwd(lambda p: f(replace(fiducial, **p)))(varied)`.
- Constructors cannot check values, because fields may be tracers. Bounds checking is an eager `check()` call.
- Emulator weights are neither leaves nor meta fields. The meta field is the model name, and a cached loader supplies the weights as jit constants.

## Protocols

The contract is two protocols that list behaviour only. Which parameters exist is the business of each implementation, so consumers never read `cosmo.h` or `cosmo.omega_b`.

The test for a member is whether it is meaningful for every cosmology. `hubble_constant` is; `w0` is not.

### Background

```python
class Background(Protocol):
    # required
    @property
    def hubble_constant(self) -> Array: ...  # H0, 1/Mpc
    def hubble_parameter(self, z) -> Array: ...  # H(z), 1/Mpc
    def matter_density_parameter(self, z) -> Array: ...  # Omega_m(z)
    def comoving_distance(self, z) -> Array: ...  # Mpc
    def redshift_at_comoving_distance(self, chi) -> Array: ...

    # provided, in terms of the above
    def transverse_comoving_distance(self, z):
        return self.comoving_distance(z)  # flat default

    def luminosity_distance(self, z):
        return (1 + z) * self.transverse_comoving_distance(z)

    def angular_diameter_distance(self, z):
        return self.transverse_comoving_distance(z) / (1 + z)

    def differential_comoving_volume(self, z):  # Mpc^3 / sr
        return self.transverse_comoving_distance(z) ** 2 / self.hubble_parameter(z)
```

The three derived quantities go through `transverse_comoving_distance`, so a curved model overrides one method and the rest stay correct.

An implementation gets the provided methods by inheriting from `Background` explicitly. A class that only matches the protocol structurally must write them itself, so inheriting is the documented way to implement a backend.

### Perturbations

```python
class Perturbations(Protocol):
    @property
    def background(self) -> Background: ...
    def growth_function(self, z, *, k=None) -> Array: ...  # D, equal to 1 at z = 0
    def growth_rate(self, z, *, k=None) -> Array: ...  # dlnD / dlna
    def matter_power_spectrum(self, k, z, *, nonlinear=False) -> Array: ...  # Mpc^3
    def primordial_power_spectrum(self, k) -> Array: ...  # dimensionless
```

- `background` lets one object satisfy both protocols by returning itself, or lets an emulator adapter pair with any native background.
- `k` on the growth methods is keyword-only, which avoids clashing with the `(k, z)` order of the power spectrum.
- `k=None` means the large-scale limit, with the shape of `z`. With `k` given, the result broadcasts `k` against `z`. Scale-independent implementations accept `k` and broadcast the same value over it.
- The scale-dependent growth function is defined as the square root of the ratio of linear spectra:

```latex
D(z, k) = \sqrt{\frac{P_{\mathrm{lin}}(k, z)}{P_{\mathrm{lin}}(k, 0)}}
```

- The contract does not say whether the spectrum factorises into a function of `k` times a function of `z`. Linear spectra without massive neutrinos do; nonlinear ones do not. Consumers that need the factorised form test for it themselves.

### Not in the protocols

- **Redshift of decoupling.** Its fitting formula needs the baryon density, which a background such as `FlatLCDM(h, omega_m)` does not have. It is a method on the implementations that can compute it, and corrtracers' CMB lensing tracer asks for it only when used.
- **"Is a pytree whose leaves are its parameters."** A `typing.Protocol` cannot express this. It is a documented requirement, checked by the conformance tests.

## Parameters

The built-in implementations take typed fields, not a parameter dict. Because the protocols carry no parameters, this is a choice per implementation and a dict-backed one can still satisfy the contract.

|  | Typed fields | Parameter dict |
| --- | --- | --- |
| Typos | Unknown argument raises at construction | Silent when a key is read with a default |
| Discoverability | Fields, defaults and docstrings in one place | Read the backend source |
| Extending | New component or new class | Add a key |
| Choosing what varies | Every field is a leaf | Only the keys passed are leaves |
| Alternative bases | Needs alternative constructors | Runtime "exactly one of" checks |
| Pytree structure | Fixed | Changes with the key set, which retraces |

A dict still works at the edge: `Cosmology(**params)` raises on unknown keys.

### Extending by composition

Physics that varies between models gets its own small typed pytree, held in a slot. Adding a model means writing one short dataclass that satisfies the slot's protocol; the cosmology class is not subclassed.

```python
cosmo = Cosmology(
    h=0.6736,
    omega_b=0.02237,
    omega_cdm=0.1200,
    dark_energy=W0Wa(w0=-1.0, wa=0.0),  # or Lambda(), or your own
    primordial=PowerLaw(ln10As=3.044, n_s=0.9649),
    power=Bacco("Arico2023"),  # static, hashable
)
```

- Parameters of a component appear in path names as `cosmo.dark_energy.w0`.
- A backend states what it supports by which components it accepts. The bacco adapter can reject an unsupported dark-energy model at construction.
- Subclassing remains possible for one-off cases. It requires re-registering the subclass with `register_dataclass` and making fields keyword-only.

### Consequences

- **Sampling in another basis** (sigma8, Omega\_m) needs a constructor such as `Cosmology.from_sigma8(...)` or a conversion before construction.
- **`fNL` leaves the cosmology.** Only corrtracers' scale-dependent bias uses it, so it belongs in corrtracers' own parameters or in a `primordial` component.
- **No `extra` dict.** It would bring back the silent-typo problem. It can be added later if a concrete case needs it.

## Conventions

One unit system and one shape rule apply to every implementation, with no `units=` arguments.

| Topic | Rule |
| --- | --- |
| Distances | Mpc |
| Wavenumbers | 1/Mpc, never h/Mpc. Adapters convert |
| Hubble rate | 1/Mpc (c = 1) |
| Power spectrum | Mpc^3 |
| Argument order | `(k, z)` whenever both are positional |
| Shapes | One cosmology per object. `k` and `z` broadcast elementwise. Batches use `jax.vmap` |
| Precision | The package never sets `jax_enable_x64`; it follows the user's configuration |
| Docstrings | NumPy style, mathematics in RST `:math:` |

**Validity.** Each implementation carries static metadata for the wavenumber range, redshift range and parameter bounds it is trusted in. `cosmo.check()` compares concrete values against them and raises; it runs outside `jit`, like `check_params` in baccoemu-jax.

**Extrapolation.** An implementation documents one behaviour outside its trusted range and does not extrapolate silently. Extending a transfer function beyond that range for an integral is the consumer's job, as corrtracers does today in `CosmoState.T_delta`.

## Backends and packaging

The implementations I maintain live in the package. Native JAX ones install with the core, and each emulator adapter sits behind an optional extra.

&#91;embedded content: package dependencies · 2 consumers, 3 implementations, 2 optional emulators\]

Consumers see only the protocols. An emulator package is imported the first time its adapter is used, so a core install never needs one.

```
the_package/
  protocols.py          # Background, Perturbations
  background.py         # FlatLCDM and other native backgrounds
  eisenstein_hu.py      # pure JAX, no extra
  backends/
    cosmopower.py       # needs [cosmopower]
    bacco/              # JAX port of the bacco emulators, needs [bacco]
  testing.py            # conformance suite
```

```toml
[project.optional-dependencies]
cosmopower = ["cosmopower-jax>=0.5.8"]
bacco = ["baccoemu"]          # only to download and load the weights
```

### Rules

- **The core depends on JAX only.** The gravitational-wave use case never pulls in an emulator.
- **Nothing under `backends/` is imported at top level.** A missing dependency raises an `ImportError` that names the extra.
- **Weights load on first use**, behind a cache, never at import.
- **A backend is a power-spectrum provider, not a full cosmology.** It plugs into the package's own class and reuses a native background, so each adapter is parameter mapping, unit conversion and a redshift table. The bacco backend is larger: it also holds the network evaluation, moved in from baccoemu-jax, which is then retired.
- **One conformance suite**, parametrised over backends and skipped when an extra is missing. It checks shapes, units, a finite `jacfwd`, and rough agreement with Eisenstein-Hu. CI runs one job per extra.
- **No registry or entry points.** The protocols are structural, so a third party can ship a backend from their own package.

### To sort out first

- [ ] Move the baccoemu-jax code and tests into `backends/bacco`, then archive that repository. The extra depends on `baccoemu` from the index, not on the local editable path used today.
- [ ] Check what `baccoemu` and `cosmopower-jax` install transitively, and whether either constrains the JAX version.
- [ ] Read the redshift and wavenumber bounds of the bacco emulators, and define the fallback for tracers beyond them.

## corrtracers migration

corrtracers drops its `cosmology/` subpackage and takes a `Perturbations` object in its place. The work is mostly removing direct reads of input parameters.

| Today in `CosmoState` | Used for | After |
| --- | --- | --- |
| `H(z)`, `chi(z)`, `z_of_chi` | Windows, Limber | `Background` methods |
| `H0`, `omega_m` | Weyl prefactor | `hubble_constant`, `matter_density_parameter(0)` |
| `D(z)`, `f(z)` | Redshift-space distortions, velocity | `growth_function`, `growth_rate` |
| `pk(cosmo, k)` at z = 0 | Every transfer function | `matter_power_spectrum(k, z)` |
| `A_s`, `n_s`, `k_pivot` | Density transfer for the fNL bias | `primordial_power_spectrum(k)` |
| `chi_star` from `omega_b` | CMB lensing source distance | Optional decoupling method, or a fixed redshift on the tracer |
| `fNL` read from the cosmology dict | Scale-dependent bias | A corrtracers parameter |

Other changes:

- **Model checks capabilities at construction.** It requires `Perturbations`, and the FFTLog integrator additionally requires a spectrum that factorises. Limber accepts any.
- **Per-parameter arrays become locals.** `Model.cl` evaluates the cosmology once on its static grids, as in the object-model section.
- **Transfer extrapolation stays in corrtracers.** The low-k and high-k rules in `T_delta` are a consumer choice about its own integration range.
- **Parameter names.** The cosmology sits in the parameter tree as an object, so names become `cosmo.h` and `cosmo.dark_energy.w0`. `path_names` already formats attribute keys.
- **During 0.x, corrtracers resolves the package from a local path** with a uv source, so the contract can change without a release each time.

## Open questions

- [x] **Package name.** `cosmograd`.
- [ ] **One `Cosmology` class or one per parameter family?** The current lean is one class with component slots for perturbation work, plus small separate classes such as `FlatLCDM` for background-only use.
- [ ] **Where `primordial_power_spectrum` lives.** It sits in `Perturbations` because only two protocols exist, yet only the fNL bias needs it. Tabulated or emulator-only implementations would have to accept an amplitude and tilt just to provide it.
- [ ] **Neutrino mass.** It is a static setting in corrtracers today and an emulator input in bacco. Making it a traced field means the native background must handle it consistently with the emulator.
- [ ] **Which matter the spectrum describes.** bacco returns the cold-matter (CDM plus baryons) spectrum; CosmoPower and Eisenstein-Hu return total matter. The contract needs either one definition or a keyword.
- [ ] **Redshift of decoupling.** Decided: it stays outside the protocols, as an optional method on implementations that have the baryon density.
- [ ] **Internal grid.** Resolution and range of the grid behind `comoving_distance` and its inverse, and whether they are meta fields or module constants.
- [ ] **When to cut the first release.** Changing the growth signature or the spectrum definition afterwards would break consumers.

## Prior art

Both reference libraries support Boltzmann codes alongside JAX, which is where most of their size comes from. A JAX-only package can take their structure and skip that machinery.

| Library | Taken | Left out |
| --- | --- | --- |
| [cloelib](https://github.com/cloe-org/cloelib/tree/main/cloelib/cosmology) (Euclid) | Separate `Background` and `Perturbations` protocols, with perturbations holding a background. Verbose method names | Input parameters as required protocol properties. `units` string arguments. One class per backend combination, built from mixins |
| [cosmoprimo](https://github.com/cosmodesi/cosmoprimo/tree/main/cosmoprimo) (DESI) | Capability by section: an engine provides only some of background, primordial, Fourier and so on | String engine registry. Attribute delegation through `__getattr__`. The parameter alias and conflict system |

I read `cloelib/cosmology/cosmology.py` and `jax_cosmology.py`, and outlined `cosmoprimo/cosmology.py`, on the date above.
