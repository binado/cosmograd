"""Physical constants used across implementations."""

C_KM_S: float = 299792.458
"""Speed of light in km/s."""

T_CMB_FIDUCIAL: float = 2.7255
"""Fiducial CMB temperature in K (Fixsen 2009)."""

OMEGA_GAMMA_H2_FIDUCIAL: float = 2.4729753e-5
"""Photon density :math:`\\Omega_\\gamma h^2` at :data:`T_CMB_FIDUCIAL`."""

NEUTRINO_TO_PHOTON_PER_SPECIES: float = 7 / 8 * (4 / 11) ** (4 / 3)
"""Energy density of one massless neutrino species relative to photons."""

OMEGA_B_FIDUCIAL: float = 0.04897
"""Fiducial baryon density parameter :math:`\\Omega_b` (Planck 2018)."""

N_S_FIDUCIAL: float = 0.9665
"""Fiducial scalar spectral index (Planck 2018)."""

LN10AS_FIDUCIAL: float = 3.047
"""Fiducial :math:`\\ln(10^{10} A_s)` at the pivot 0.05/Mpc (Planck 2018)."""
