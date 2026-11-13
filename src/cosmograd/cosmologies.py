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
Massless neutrinos only. The astropy realizations include one 0.06 eV massive
neutrino, which is not modelled here, so distances agree with astropy to about 0.1% (for ``Planck18`` at ``z <= 3``) rather than
exactly. ``omega_m`` is the total matter density, as in astropy.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from .background import Background, FlatLCDM

__all__ = [
    "Planck13",
    "Planck15",
    "Planck18",
    "available",
]

# Planck Collaboration 2014, A&A, 571, A16 (Paper XVI),
# arXiv:1303.5076. Table 5 (Planck + WP + highL + BAO).
Planck13: Final = FlatLCDM(H0=67.77, omega_m=0.30712, n_eff=3.046)
"""Planck 2013."""
# Planck Collaboration 2016, A&A, 594, A13 (Paper XIII),
# arXiv:1502.01589. Table 4 (TT, TE, EE + lowP + lensing + ext).
Planck15: Final = FlatLCDM(H0=67.74, omega_m=0.3075, n_eff=3.046)
"""Planck 2015."""
# Planck Collaboration 2020, A&A, 641, A6 (Paper VI),
# arXiv:1807.06209. Table 2 (TT, TE, EE + lowE + lensing + BAO).
Planck18: Final = FlatLCDM(H0=67.66, omega_m=0.30966, n_eff=3.046)
"""Planck 2018."""

available: Final[Mapping[str, Background]] = MappingProxyType(
    {
        "Planck13": Planck13,
        "Planck15": Planck15,
        "Planck18": Planck18,
    }
)
"""Read-only registry of the constants above by name, for iteration and
configuration. It is typed as :class:`~cosmograd.Background`; import the
constants directly to keep the concrete type."""
