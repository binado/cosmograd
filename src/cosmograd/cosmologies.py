"""Standard cosmologies, mirroring :mod:`astropy.cosmology.realizations`.

Each name is a module-level constant with its concrete class, so attribute
access, autocompletion and :meth:`~cosmograd.FlatLCDM.with_parameters` are
checked statically. The instances are immutable pytrees and safe to share.

Background parameters follow the corresponding astropy realizations (``H0``,
``Tcmb0``, ``Neff``, ``Ob0``), ``n_s`` and ``ln10as`` come from the same table
column, and all are stored as Python floats, so importing this module does not
initialise JAX and the precision follows ``jax_enable_x64``.

Notes
-----
There are no massive neutrinos here, so ``omega_m`` is the total
non-relativistic density including the neutrinos, as quoted in the papers. It
is astropy's ``Om0 + Onu0``: astropy's ``Om0`` excludes massive neutrinos, which
it tracks separately in ``Onu0``. Comoving distances then agree with astropy to
about 5e-5 for ``z <= 3`` (the neutrino is treated as matter at all redshifts).
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
# https://arxiv.org/abs/1303.5076. Table 5 (Planck + WP + highL + BAO).
# omega_m = astropy Om0 (0.30712) + Onu0.
Planck13: Final = FlatLCDM(
    H0=67.77, omega_m=0.30856, omega_b=0.048252, n_s=0.9611, ln10as=3.0973, n_eff=3.046
)
"""Planck 2013."""
# Planck Collaboration 2016, A&A, 594, A13 (Paper XIII),
# https://arxiv.org/abs/1502.01589. Table 4 (TT, TE, EE + lowP + lensing + ext).
# omega_m = astropy Om0 (0.3075) + Onu0.
Planck15: Final = FlatLCDM(
    H0=67.74, omega_m=0.30894, omega_b=0.0486, n_s=0.9667, ln10as=3.064, n_eff=3.046
)
"""Planck 2015."""
# Planck Collaboration 2020, A&A, 641, A6 (Paper VI),
# https://arxiv.org/abs/1807.06209. Table 2 (TT, TE, EE + lowE + lensing + BAO).
# omega_m = astropy Om0 (0.30966) + Onu0.
Planck18: Final = FlatLCDM(
    H0=67.66, omega_m=0.31110, omega_b=0.04897, n_s=0.9665, ln10as=3.047, n_eff=3.046
)
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
