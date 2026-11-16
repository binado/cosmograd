"""JAX port of the bacco emulators. Needs the ``bacco`` extra.

The native ``baccoemu`` networks evaluate in float32 using JAX. Weights
are downloaded and models loaded on first use.
"""

from ._cosmology import BaccoEmu, BaccoEmuParameters
from ._emulator import load_baccoemu

__all__ = ["BaccoEmu", "BaccoEmuParameters", "load_baccoemu"]
