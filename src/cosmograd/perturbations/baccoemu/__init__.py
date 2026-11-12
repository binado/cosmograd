"""JAX port of the bacco emulators. Needs the ``bacco`` extra.

The network evaluation is pure JAX; ``baccoemu`` itself is used only to
download and load the weights, on first use.
"""

from ._cosmology import BaccoEmu
from ._emulator import load_baccoemu

__all__ = ["BaccoEmu", "load_baccoemu"]
