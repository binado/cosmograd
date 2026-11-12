"""Eisenstein–Hu (1998) linear power with baryon acoustic oscillations.

References
----------
.. [EH98] Eisenstein, D. J. & Hu, W. (1998), "Baryonic Features in the Matter
   Transfer Function", ApJ 496, 605–614. https://doi.org/10.1086/305424
   Open manuscript: https://arxiv.org/abs/astro-ph/9709112
.. [Heath77] Heath, D. J. (1977), "The growth of density perturbations in zero
   pressure Friedmann–Lemaître universes", MNRAS 179, 351–358.
   https://doi.org/10.1093/mnras/179.3.351
.. [Hamilton01] Hamilton, A. J. S. (2001), "Formulae for growth factors in
   expanding universes containing matter and a cosmological constant",
   MNRAS 322, 419–425. https://arxiv.org/abs/astro-ph/0006089
.. [SmithAngulo19] Smith, R. E. & Angulo, R. E. (2019), "Precision modelling of
   the matter power spectrum in a Planck-like Universe", MNRAS 486, 1448–1479.
   https://arxiv.org/abs/1807.00040
.. [Planck20] Planck Collaboration (2020), "Planck 2018 results. VI.
   Cosmological parameters", A&A 641, A6, section 2.1.
   https://arxiv.org/abs/1807.06209
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from ..background import FlatLCDM
from ..utils import (
    affine_transform_gauss_legendre_nodes,
    gauss_legendre_nodes_and_weights,
)

__all__ = ["EisensteinHu"]


def _t0(
    q: Real[ArrayLike, "*shape"], alpha: Real[ArrayLike, ""], beta: Real[ArrayLike, ""]
) -> Float[Array, "*shape"]:
    """Zero-baryon building block, [EH98]_ equations (19)–(20)."""
    c = 14.2 / alpha + 386.0 / (1.0 + 69.9 * q**1.08)
    lg = jnp.log(jnp.e + 1.8 * beta * q)
    return lg / (lg + c * q**2)


def _transfer(
    k: Real[ArrayLike, "*shape"],
    omega_b: Real[ArrayLike, ""],
    omega_m_h2: Real[ArrayLike, ""],
    t_cmb: float,
) -> Float[Array, "*shape"]:
    r"""Matter transfer function ``T(k)`` of EH98 (:math:`\to 1` as :math:`k \to 0`).

    Parameters
    ----------
    k
        Wavenumbers in 1/Mpc.
    omega_b
        Physical baryon density :math:`\Omega_b h^2`.
    omega_m_h2
        Physical matter density :math:`\Omega_m h^2`.
    t_cmb
        CMB temperature in K.
    """
    wm, wb = omega_m_h2, omega_b
    theta = t_cmb / 2.7
    fb = wb / wm
    fc = 1.0 - fb

    # EH98 equations (2)–(5): equality, drag epoch and baryon/photon ratio.
    z_eq = 2.50e4 * wm * theta**-4
    k_eq = 7.46e-2 * wm * theta**-2
    b1 = 0.313 * wm**-0.419 * (1.0 + 0.607 * wm**0.674)
    b2 = 0.238 * wm**0.223
    z_d = 1291.0 * wm**0.251 / (1.0 + 0.659 * wm**0.828) * (1.0 + b1 * wb**b2)

    def r_ratio(z: Real[ArrayLike, ""]) -> Float[Array, ""]:
        return jnp.asarray(31.5 * wb * theta**-4 * (1000.0 / z))

    r_d, r_eq = r_ratio(z_d), r_ratio(z_eq)
    # EH98 equations (6)–(7): sound horizon and Silk damping scale.
    s = (
        2.0
        / (3.0 * k_eq)
        * jnp.sqrt(6.0 / r_eq)
        * jnp.log((jnp.sqrt(1.0 + r_d) + jnp.sqrt(r_d + r_eq)) / (1.0 + jnp.sqrt(r_eq)))
    )
    k_silk = 1.6 * wb**0.52 * wm**0.73 * (1.0 + (10.4 * wm) ** -0.95)

    q = k / (13.41 * k_eq)
    ks = k * s

    # EH98 equations (10)–(12), (17)–(20): CDM suppression and interpolation.
    a1 = (46.9 * wm) ** 0.670 * (1.0 + (32.1 * wm) ** -0.532)
    a2 = (12.0 * wm) ** 0.424 * (1.0 + (45.0 * wm) ** -0.582)
    alpha_c = a1**-fb * a2 ** (-(fb**3))
    b1c = 0.944 / (1.0 + (458.0 * wm) ** -0.708)
    b2c = (0.395 * wm) ** -0.0266
    beta_c = 1.0 / (1.0 + b1c * (fc**b2c - 1.0))
    f = 1.0 / (1.0 + (ks / 5.4) ** 4)
    t_c = f * _t0(q, 1.0, beta_c) + (1.0 - f) * _t0(q, alpha_c, beta_c)

    # EH98 equations (14)–(15), (21)–(24): baryon envelope and acoustic nodes.
    y = (1.0 + z_eq) / (1.0 + z_d)
    sqy = jnp.sqrt(1.0 + y)
    g = y * (-6.0 * sqy + (2.0 + 3.0 * y) * jnp.log((sqy + 1.0) / (sqy - 1.0)))
    alpha_b = 2.07 * k_eq * s * (1.0 + r_d) ** -0.75 * g
    beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * jnp.sqrt((17.2 * wm) ** 2 + 1.0)
    beta_node = 8.41 * wm**0.435
    s_tilde = s / (1.0 + (beta_node / ks) ** 3) ** (1.0 / 3.0)
    x = k * s_tilde
    j0 = jnp.sinc(x / jnp.pi)
    t_b = (
        _t0(q, 1.0, 1.0) / (1.0 + (ks / 5.2) ** 2)
        + alpha_b / (1.0 + (beta_b / ks) ** 3) * jnp.exp(-((k / k_silk) ** 1.4))
    ) * j0
    # EH98 equation (16): density-weighted matter transfer function.
    return fb * t_b + fc * t_c


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EisensteinHu:
    """EH98 linear perturbations on a supplied flat ΛCDM background.

    Parameters
    ----------
    background : FlatLCDM
        Background object, retained by identity. Its Hubble constant, matter
        density and CMB temperature set the transfer function.
    omega_b : ArrayLike
        Baryon density today, Ω_b (not Ω_b h²), with 0 < Ω_b < Ω_m.
    n_s : ArrayLike
        Primordial scalar spectral index.
    ln10as : ArrayLike
        Natural logarithm of 10¹⁰ A_s, at the pivot k = 0.05/Mpc.

    Notes
    -----
    Growth neglects radiation and uses 32-point Gauss–Legendre quadrature,
    checked for representative late-time cosmologies through z = 10.
    Background distances retain the supplied background's radiation treatment.
    Massive neutrinos and nonlinear power are unsupported. Inputs are not
    validated: wavenumbers must be positive and redshifts non-negative;
    neither needs to be sorted.

    Only the background parameters and three spectral parameters are pytree
    leaves. Fields may hold batched parameters or Jacobians; evaluate one
    cosmology at a time, using :func:`jax.vmap` for batches.
    """

    __no_type_check__ = True

    background: FlatLCDM
    omega_b: Real[ArrayLike, "..."]
    n_s: Real[ArrayLike, "..."]
    ln10as: Real[ArrayLike, "..."]

    def _growth(
        self, z: Real[ArrayLike, "*shape"]
    ) -> tuple[Float[Array, "*shape"], Float[Array, "*shape"]]:
        r"""Return matter-era-normalised growth and its logarithmic rate.

        The growing-mode integral of [Heath77]_, written with the normalisation
        of [Hamilton01]_ equations (1)–(2), is

        .. math::

            D_{\rm md}(a) = \frac{5\Omega_m}{2} E(a) I(a), \qquad
            I(a) = \int_0^a \frac{da'}{[a'E(a')]^3}, \qquad
            E(a)^2 = \Omega_m a^{-3} + 1 - \Omega_m.

        Here a = 1 today, and D_md approaches a in matter domination.
        Hamilton instead sets a = 1 at the epoch of interest. The quadrature
        substitution a' = a u² is an implementation choice. Differentiating
        this same integral gives the rate, without finite differences:

        .. math::

            f(a) = -\frac{3\Omega_m a^{-3}}{2E(a)^2}
                   + \frac{1}{a^2 E(a)^3 I(a)}.
        """
        with jax.ensure_compile_time_eval():
            nodes, weights = gauss_legendre_nodes_and_weights(32)
            u = affine_transform_gauss_legendre_nodes(nodes, 0.0, 1.0)
        a = 1 / (1 + jnp.asarray(z))
        om = jnp.asarray(self.background.omega_m)
        # a' = a u² turns the integrand into a smooth function proportional
        # to u⁴, including at the matter-era endpoint a' = 0.
        ap = a[..., None] * u**2
        integral = a**2.5 * jnp.sum(
            weights * u**4 / (om + (1 - om) * ap**3) ** 1.5, axis=-1
        )
        e2 = om * a**-3 + 1 - om
        e = jnp.sqrt(e2)
        d_md = 2.5 * om * e * integral
        rate = -1.5 * om * a**-3 / e2 + 1 / (a**2 * e**3 * integral)
        return d_md, rate

    def growth_function(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Return scale-independent growth normalised to unity today.

        Parameters
        ----------
        z : ArrayLike
            Non-negative redshift.
        k : ArrayLike, optional
            Positive wavenumber in 1/Mpc; affects broadcasting only.

        Returns
        -------
        Array
            Dimensionless growth, with the broadcast shape of z and k.
        """
        if k is not None:
            z, _ = jnp.broadcast_arrays(z, k)
        return self._growth(z)[0] / self._growth(0.0)[0]

    def growth_rate(
        self,
        z: Real[ArrayLike, "*#shape"],
        *,
        k: Real[ArrayLike, "*#shape"] | None = None,
    ) -> Float[Array, "*shape"]:
        """Return the scale-independent rate d ln D / d ln a.

        Parameters
        ----------
        z : ArrayLike
            Non-negative redshift.
        k : ArrayLike, optional
            Positive wavenumber in 1/Mpc; affects broadcasting only.

        Returns
        -------
        Array
            Dimensionless rate, with the broadcast shape of z and k.
        """
        if k is not None:
            z, _ = jnp.broadcast_arrays(z, k)
        return self._growth(z)[1]

    def matter_power_spectrum(
        self,
        k: Real[ArrayLike, "*#shape"],
        z: Real[ArrayLike, "*#shape"],
        *,
        nonlinear: bool = False,
    ) -> Float[Array, "*shape"]:
        """Return the primordial-amplitude-normalised linear matter spectrum.

        Parameters
        ----------
        k : ArrayLike
            Positive wavenumber in 1/Mpc.
        z : ArrayLike
            Non-negative redshift; broadcasts elementwise against k.
        nonlinear : bool, default False
            Nonlinear power is unsupported.

        Returns
        -------
        Array
            Power in Mpc³, with the broadcast shape of k and z.

        Notes
        -----
        [SmithAngulo19]_ equation (10), with their a g = D_md and c = 1,
        gives P = (2π²/k³) P_R [2 k² T D_md / (5 Ω_m H0²)]².
        H0 is in inverse Mpc. The growth normalised to unity today cannot be
        substituted for D_md without changing the primordial normalisation.

        Raises
        ------
        NotImplementedError
            If nonlinear power is requested.
        """
        # Smith & Angulo (2019), equation (10), with c = 1 and a g = D_md.
        # D_md must retain its matter-era normalisation for the A_s conversion.
        if nonlinear:
            raise NotImplementedError("EisensteinHu supports linear power only")
        k = jnp.asarray(k)
        h2 = (jnp.asarray(self.background.H0) / 100) ** 2
        om = jnp.asarray(self.background.omega_m)
        transfer = _transfer(k, self.omega_b * h2, om * h2, self.background.t_cmb)
        delta = (
            0.4
            * k**2
            * transfer
            * self._growth(z)[0]
            / (om * self.background.hubble_constant**2)
        )
        return 2 * jnp.pi**2 / k**3 * self.primordial_power_spectrum(k) * delta**2

    def primordial_power_spectrum(
        self, k: Real[ArrayLike, "*shape"]
    ) -> Float[Array, "*shape"]:
        """Return dimensionless primordial curvature power.

        Parameters
        ----------
        k : ArrayLike
            Positive wavenumber in 1/Mpc.

        Returns
        -------
        Array
            A_s (k / k_pivot) ** (n_s - 1), with the shape of k.

        Notes
        -----
        [SmithAngulo19]_ equation (8) with zero running. The pivot 0.05/Mpc
        and logarithmic amplitude parameter follow [Planck20]_ section 2.1:
        A_s = 10⁻¹⁰ exp(ln10as). P_R here is dimensionless (their Δ_R²).
        """
        return 1e-10 * jnp.exp(self.ln10as) * (jnp.asarray(k) / 0.05) ** (self.n_s - 1)
