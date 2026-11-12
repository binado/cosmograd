"""Test configuration. The package itself never changes JAX precision."""

import jax

jax.config.update("jax_enable_x64", True)
