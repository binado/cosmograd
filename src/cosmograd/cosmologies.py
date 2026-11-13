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

# Spergel et al. 2003, ApJS, 148, 175, doi:10.1086/377226. Table 7
# (WMAP + CBI + ACBAR + 2dFGRS + Lyman-alpha).
WMAP1: Final = FlatLCDM(H0=72.0, omega_m=0.257, t_cmb=2.725, n_eff=3.04)
"""WMAP first-year."""
# Spergel et al. 2007, ApJS, 170, 377, doi:10.1086/513700. Table 6
# (WMAP + SNGold).
WMAP3: Final = FlatLCDM(H0=70.1, omega_m=0.276, t_cmb=2.725, n_eff=3.04)
"""WMAP three-year."""
# Komatsu et al. 2009, ApJS, 180, 330, doi:10.1088/0067-0049/180/2/330.
# Table 1 (WMAP + BAO + SN, maximum likelihood).
WMAP5: Final = FlatLCDM(H0=70.2, omega_m=0.277, t_cmb=2.725, n_eff=3.04)
"""WMAP five-year."""
# Komatsu et al. 2011, ApJS, 192, 18, doi:10.1088/0067-0049/192/2/18.
# Table 1 (WMAP + BAO + H0, maximum likelihood).
WMAP7: Final = FlatLCDM(H0=70.4, omega_m=0.272, t_cmb=2.725, n_eff=3.04)
"""WMAP seven-year."""
# Hinshaw et al. 2013, ApJS, 208, 19, doi:10.1088/0067-0049/208/2/19.
# Table 4 (WMAP9 + eCMB + BAO + H0, last column).
WMAP9: Final = FlatLCDM(H0=69.32, omega_m=0.2865, t_cmb=2.725, n_eff=3.04)
"""WMAP nine-year."""
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
