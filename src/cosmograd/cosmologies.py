"""Standard cosmologies, mirroring :mod:`astropy.cosmology.realizations`.

Each name is a module-level constant with its concrete class, so attribute
access, autocompletion and :meth:`~cosmograd.FlatLCDM.with_parameters` are
checked statically. The instances are immutable pytrees and safe to share.

Parameters are those of the corresponding astropy realizations (``H0``,
``Om0``, ``Tcmb0``, ``Neff``). They are stored as Python floats so importing
this module does not initialise JAX and the precision follows
``jax_enable_x64``.

Notes
-----
Massless neutrinos only. The Planck realizations in astropy include one
0.06 eV massive neutrino, which is not modelled here, so Planck distances
agree with astropy to about 0.1% (for ``Planck18`` at ``z <= 3``) rather than
exactly. ``omega_m`` is the total matter density, as in astropy.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from .background import Background, FlatLCDM

__all__ = [
    "WMAP1",
    "WMAP3",
    "WMAP5",
    "WMAP7",
    "WMAP9",
    "Planck13",
    "Planck15",
    "Planck18",
    "available",
]

WMAP1: Final = FlatLCDM(H0=72.0, omega_m=0.257, t_cmb=2.725, n_eff=3.04)
"""WMAP first-year (Spergel et al. 2003)."""
WMAP3: Final = FlatLCDM(H0=70.1, omega_m=0.276, t_cmb=2.725, n_eff=3.04)
"""WMAP three-year (Spergel et al. 2007)."""
WMAP5: Final = FlatLCDM(H0=70.2, omega_m=0.277, t_cmb=2.725, n_eff=3.04)
"""WMAP five-year (Komatsu et al. 2009)."""
WMAP7: Final = FlatLCDM(H0=70.4, omega_m=0.272, t_cmb=2.725, n_eff=3.04)
"""WMAP seven-year (Komatsu et al. 2011)."""
WMAP9: Final = FlatLCDM(H0=69.32, omega_m=0.2865, t_cmb=2.725, n_eff=3.04)
"""WMAP nine-year (Hinshaw et al. 2013)."""
Planck13: Final = FlatLCDM(H0=67.77, omega_m=0.30712, n_eff=3.046)
"""Planck 2013 (Planck Collaboration XVI 2014), Table 2 column 2."""
Planck15: Final = FlatLCDM(H0=67.74, omega_m=0.3075, n_eff=3.046)
"""Planck 2015 (Planck Collaboration XIII 2016), Table 4 last column."""
Planck18: Final = FlatLCDM(H0=67.66, omega_m=0.30966, n_eff=3.046)
"""Planck 2018 (Planck Collaboration VI 2020), Table 2 last column."""

available: Final[Mapping[str, Background]] = MappingProxyType(
    {
        "WMAP1": WMAP1,
        "WMAP3": WMAP3,
        "WMAP5": WMAP5,
        "WMAP7": WMAP7,
        "WMAP9": WMAP9,
        "Planck13": Planck13,
        "Planck15": Planck15,
        "Planck18": Planck18,
    }
)
"""Read-only registry of the constants above by name, for iteration and
configuration. It is typed as :class:`~cosmograd.Background`; import the
constants directly to keep the concrete type."""
