"""Physical constants used across implementations."""

C_KM_S: float = 299792.458
"""Speed of light in km/s."""

T_CMB_FIDUCIAL: float = 2.7255
"""Fiducial CMB temperature in K (Fixsen 2009)."""

OMEGA_GAMMA_H2_FIDUCIAL: float = 2.4728e-5
"""Photon density :math:`\\Omega_\\gamma h^2` at :data:`T_CMB_FIDUCIAL`."""

NEUTRINO_TO_PHOTON_PER_SPECIES: float = 7 / 8 * (4 / 11) ** (4 / 3)
"""Energy density of one massless neutrino species relative to photons."""
