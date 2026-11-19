# Changelog

## Unreleased

- Added opt-in `extrapolate=True` (and `slope_fraction`) to `CosmoPower` and
  `BaccoEmu`: the linear spectrum, and growth read off it, continue smoothly
  outside the emulator's k range, as `k P_R(k)` below `k_min` and with the
  measured log-log slope above `k_max`. Added `k_min`/`k_max` properties. Defaults are unchanged.

## 0.2.0

- Added `cosmograd.background.functional`: distances and the matter density as
  pure functions of a `hubble_parameter(params, z)` callable and any pytree of
  parameters. The native backgrounds now delegate to them.
- Added `GenericBackground`, a `Background` over such a callable and parameters
  in the caller's own layout (dict, named tuple, dataclass).
- `EisensteinHu`, `CosmoPower` and `BaccoEmu` now accept any `Background` of
  the expansion history they model. A `parameters` accessor supplies what the
  background cannot (baryon density, primordial parameters and, per backend,
  the CMB temperature or `w0` and `wa`), and defaults to reading the
  cosmology's attributes.
- `EisensteinHu` no longer removes radiation from the background for growth.
  The growth ODE starts at `a_initial`, by default where the matter fraction
  peaks (`matter_domination_scale_factor`), and growth is the scale factor
  before that. Power spectra shift by about 0.1%.
- `linear_growth` accepts a traced `a_initial`.

## 0.1.1

- Simplified the README and added this changelog.

## 0.1.0

- Initial release of differentiable cosmology models in JAX.
- Added flat and curved LCDM and w0waCDM backgrounds, distances and growth.
- Added Eisenstein–Hu, CosmoPower and bacco matter power spectrum backends.
- Added Planck cosmologies and comparisons against independent implementations.
