# Changelog

## Unreleased

- Added `cosmograd.background.functional`: distances and the matter density as
  pure functions of a `hubble_parameter(params, z)` callable and any pytree of
  parameters. The native backgrounds now delegate to them.

## 0.1.1

- Simplified the README and added this changelog.

## 0.1.0

- Initial release of differentiable cosmology models in JAX.
- Added flat and curved LCDM and w0waCDM backgrounds, distances and growth.
- Added Eisenstein–Hu, CosmoPower and bacco matter power spectrum backends.
- Added Planck cosmologies and comparisons against independent implementations.
