r"""
Spectral-moment sum rules for the nonequilibrium Anderson impurity model,
spin-resolved.

The retarded impurity GF admits the exact large-frequency expansion

.. math::

    G^R_\sigma(\omega) = \sum_m \frac{M_m^\sigma}{\omega^{m+1}}, \qquad
    M_m^\sigma = \big\langle \{ \mathcal{L}^m f_\sigma,\, f_\sigma^\dagger \} \big\rangle,

with :math:`\mathcal{L}X = [X, H]`. The nested-commutator form is a
state-independent operator identity, so the moments hold in any steady
state, in or out of equilibrium (the expectation values are taken in the
actual NESS). Writing

.. math::

    D_k^\sigma = -\frac{1}{\pi}\int d\omega\, \omega^{k-1}\, \mathrm{Im}\,\Delta^R_\sigma(\omega)

for the coefficients of
:math:`\Delta^R_\sigma(\omega) = \sum_k D_k^\sigma/\omega^k`, the first
three read

.. math::

    M_1^\sigma &= \epsilon_\sigma + U n_{\bar\sigma} \\
    M_2^\sigma &= (M_1^\sigma)^2 + U^2 n_{\bar\sigma}(1-n_{\bar\sigma}) + D_1^\sigma \\
    M_3^\sigma &= \epsilon_\sigma^3 + U n_{\bar\sigma}\big(3\epsilon_\sigma^2 + 3\epsilon_\sigma U + U^2\big)
                  + 2 D_1^\sigma M_1^\sigma + D_2^\sigma + U^2 C_{\bar\sigma}

:math:`M_1` and :math:`M_2` are the spin-resolved form of the sum rules
already used in the preprint; both close on the occupations alone, so they
are external benchmarks.

:math:`M_3` does not. Its last term is the two-particle correlator

.. math::

    C_{\bar\sigma} = \mathrm{Re}\big\langle (2 n_\sigma - 1) A^\dagger_{\bar\sigma}\big\rangle,
    \qquad A^\dagger_{\bar\sigma} = \sum_{\lambda p} t_{\lambda\bar\sigma}\, c^\dagger_{\lambda p \bar\sigma} f_{\bar\sigma}

i.e. the nonequilibrium Keldysh analogue of the Potthoff-Wegner-Nolting
[Phys. Rev. B 55, 16132 (1997)] :math:`m=3` band shift, and equal to
:math:`n_{\bar\sigma}(1-n_{\bar\sigma})(\tilde B_{\bar\sigma} - \epsilon_{\bar\sigma})`.
Note the flavor index: the :math:`\sigma` moment needs the *opposite* spin's
band shift -- the same ``B_tilde[other]`` that
:meth:`Solver.calc_SE <neq_kk_ipt_solver.Solver.calc_SE>` already consumes.

Using :math:`\Sigma_\sigma G_\sigma = U \langle\langle n_{\bar\sigma} f_\sigma ; f_\sigma^\dagger\rangle\rangle`
and Langreth on the triple product, :math:`C_{\bar\sigma}` is the functional
of the converged solution

.. math::

    C_{\bar\sigma} = \mathrm{Re}\left\{-i\int\frac{d\omega}{2\pi}
    \big[Q_{\bar\sigma} G_{\bar\sigma} \Delta_{\bar\sigma}\big]^<\right\},
    \qquad Q = \frac{2\Sigma}{U} - 1

which is exactly what
:meth:`Solver._calc_band_shift_interacting <neq_kk_ipt_solver.Solver._calc_band_shift_interacting>`
integrates. The :math:`M_3` sum rule is therefore a consistency check
between two different functionals of the same :math:`(G, \Sigma)` -- not an
absolute benchmark like :math:`M_1`/:math:`M_2` -- and is a direct probe of
whether the self-energy ansatz carries the right :math:`m=3` structure.

Author: Tommaso Maria Mazzocchi
"""
import numpy as np

from .utils import Keldysh


def _lesser(X: Keldysh) -> np.ndarray:
    r"""Lesser component, :math:`X^< = X^K/2 - i\,\mathrm{Im}\,X^R`."""
    return X.K / 2.0 - 1j * np.imag(X.R)


def hybridization_moments(w: np.ndarray, Delta_R: np.ndarray, kmax: int = 2) -> dict[int, float]:
    r"""
    Coefficients of the large-:math:`\omega` expansion of the hybridization.

    .. math::

        \Delta^R(\omega) = \sum_k \frac{D_k}{\omega^k}, \qquad
        D_k = -\frac{1}{\pi}\int d\omega\, \omega^{k-1}\, \mathrm{Im}\,\Delta^R(\omega)

    `D_1` is the total spectral weight of the hybridization (the preprint's
    :math:`D_1`) and `D_2` its first moment.

    Parameters
    ----------
    w : numpy.ndarray
        Frequency grid.
    Delta_R : numpy.ndarray of complex
        Retarded hybridization function, :math:`\Delta^R(\omega)`.
    kmax : int, default=2
        Highest-order coefficient to compute (``D_1`` through ``D_kmax``).

    Returns
    -------
    dict[int, float]
        Maps ``k`` to :math:`D_k`, for ``k`` in ``1..kmax``.
    """
    im_Delta = np.imag(Delta_R)
    return {
        k: float(-np.trapezoid(w ** (k - 1) * im_Delta, w) / np.pi)
        for k in range(1, kmax + 1)
    }


def spectral_moments(w: np.ndarray, GF_R: np.ndarray, orders=(0, 1, 2, 3)) -> dict[int, float]:
    r"""
    Numerically integrated spectral moments :math:`\int d\omega\, \omega^m A(\omega)`.

    Parameters
    ----------
    w : numpy.ndarray
        Frequency grid.
    GF_R : numpy.ndarray of complex
        Retarded Green's function, :math:`G^R(\omega)`, whose spectral
        function :math:`A(\omega) = -\mathrm{Im}\,G^R(\omega)/\pi` is
        integrated.
    orders : iterable of int, default=(0, 1, 2, 3)
        Which moment orders `m` to compute.

    Returns
    -------
    dict[int, float]
        Maps `m` to the integrated moment, for each `m` in `orders`.
    """
    A = -np.imag(GF_R) / np.pi
    return {m: float(np.trapezoid(w ** m * A, w)) for m in orders}


def band_shift_correlator(
    w: np.ndarray, GF: Keldysh, SE: Keldysh, Delta: Keldysh, U: float
) -> float:
    r"""
    The two-particle correlator :math:`C_\sigma` entering the exact :math:`M_3` sum rule.

    .. math::

        C_\sigma = \mathrm{Re}\big\langle (2 n_{\bar\sigma} - 1) A^\dagger_\sigma\big\rangle
        = \mathrm{Re}\left\{-i\int\frac{d\omega}{2\pi}\big[Q\,G\,\Delta\big]^<\right\}

    with :math:`Q = 2\Sigma/U - 1` (whose lesser component is
    :math:`2\Sigma^</U`, the :math:`-1` being instantaneous) and the
    Langreth rule

    .. math::

        [Q\,G\,\Delta]^< = Q^R G^R \Delta^< + Q^R G^< \Delta^A + Q^< G^A \Delta^A.

    Parameters
    ----------
    w : numpy.ndarray
        Frequency grid.
    GF : Keldysh
        Impurity Green's function for the flavor entering the correlator.
    SE : Keldysh
        Self-energy for the same flavor.
    Delta : Keldysh
        Hybridization function for the same flavor.
    U : float
        Hubbard interaction strength; must be nonzero.

    Returns
    -------
    float
        The correlator :math:`C_\sigma`.

    Raises
    ------
    ValueError
        If `U` is (numerically) zero.

    Notes
    -----
    Same integral as
    :meth:`Solver._calc_band_shift_interacting <neq_kk_ipt_solver.Solver._calc_band_shift_interacting>`,
    but returned raw -- without dividing by :math:`n(1-n)` -- since that is
    the combination the :math:`m=3` sum rule needs, and it stays well
    behaved as `n` approaches 0 or 1.
    """
    if abs(U) < 1e-12:
        raise ValueError("The band-shift correlator requires finite U.")

    Q_R = 2.0 * SE.R / U - 1.0
    Q_lss = 2.0 * _lesser(SE) / U

    G_lss = _lesser(GF)
    G_A = np.conj(GF.R)
    Delta_lss = _lesser(Delta)
    Delta_A = np.conj(Delta.R)

    X_lss = Q_R * GF.R * Delta_lss + Q_R * G_lss * Delta_A + Q_lss * G_A * Delta_A

    return float(np.real(-1j * np.trapezoid(X_lss, w) / (2.0 * np.pi)))


def double_occupancy(w: np.ndarray, GF: Keldysh, SE: Keldysh, U: float) -> float:
    r"""
    Keldysh Galitskii-Migdal double occupancy evaluated from one spin channel.

    .. math::

        n_d = -\frac{i}{2\pi U}\int d\omega\,
        \big[\Sigma^R(\omega) G^<(\omega) + \Sigma^<(\omega) G^A(\omega)\big]

    The same expression
    :meth:`Solver.calc_occupations <neq_kk_ipt_solver.Solver.calc_occupations>`
    uses (averaged over both flavors there), but with the flavor left to the
    caller instead of fixed.

    Parameters
    ----------
    w : numpy.ndarray
        Frequency grid.
    GF : Keldysh
        Impurity Green's function for the flavor to evaluate.
    SE : Keldysh
        Self-energy for the same flavor.
    U : float
        Hubbard interaction strength; must be nonzero.

    Returns
    -------
    float
        The double occupancy :math:`n_d`, as estimated from this one flavor.

    Raises
    ------
    ValueError
        If `U` is (numerically) zero.

    Notes
    -----
    Exactly, the result is independent of which spin is used: the equation
    of motion gives
    :math:`\Sigma_\sigma G_\sigma = U\langle\langle n_{\bar\sigma} f_\sigma ; f_\sigma^\dagger\rangle\rangle`,
    whose equal-time lesser component is :math:`U\langle n_{\bar\sigma} n_\sigma\rangle`
    either way. Evaluating it for both spins and comparing is therefore a
    free internal-consistency check on an approximate solution -- one that
    probes the two-particle sector, unlike the spectral moment sum rules
    above. It has no content when the two flavors are equivalent by
    symmetry; it only bites for spin-dependent setups.
    """
    if abs(U) < 1e-12:
        raise ValueError("The Galitskii-Migdal double occupancy requires finite U.")

    GF_A = np.conj(GF.R)
    integrand = SE.R * _lesser(GF) + _lesser(SE) * GF_A

    return float(np.real(-1j / (2.0 * np.pi * U) * np.trapezoid(integrand, w)))


def closed_form_moments(solver, fl: str) -> dict[str, float]:
    """
    Evaluate the exact :math:`M_1, M_2, M_3` sum rules for one flavor.

    Uses the converged `solver`'s onsite energies, `U`, self-consistent
    occupations, hybridization moments, and (for :math:`M_3`) the opposite
    flavor's band-shift correlator.

    Parameters
    ----------
    solver : neq_kk_ipt_solver.Solver
        A solver instance that has already converged (:meth:`Solver.solve`
        has been called successfully).
    fl : str
        Which flavor (must be one of ``solver.flavors``) to evaluate the
        moments for.

    Returns
    -------
    dict[str, float]
        Keys ``"M1"``, ``"M2"``, ``"M3"``, ``"D1"``, ``"D2"``, ``"C_bar"``,
        ``"B_tilde_bar"``, ``"n_bar"``.
    """
    other = solver._other_flavor(fl)

    U = float(solver.U)
    eps = float(solver.impurity_onsite_e[fl])
    eps_bar = float(solver.impurity_onsite_e[other])
    n_bar = float(np.real(solver.n_occ[other]))

    D = hybridization_moments(solver.w, solver.Delta[fl].R, kmax=2)

    C_bar = band_shift_correlator(
        solver.w, solver.GF[other], solver.SE[other], solver.Delta[other], U
    )

    M1 = eps + U * n_bar
    M2 = M1 ** 2 + U ** 2 * n_bar * (1.0 - n_bar) + D[1]
    M3 = (
        eps ** 3
        + U * n_bar * (3.0 * eps ** 2 + 3.0 * eps * U + U ** 2)
        + 2.0 * D[1] * M1
        + D[2]
        + U ** 2 * C_bar
    )

    denom = n_bar * (1.0 - n_bar)
    B_tilde_bar = eps_bar + C_bar / denom if denom > 1e-12 else float("nan")

    return {
        "M1": M1,
        "M2": M2,
        "M3": M3,
        "D1": D[1],
        "D2": D[2],
        "C_bar": C_bar,
        "B_tilde_bar": B_tilde_bar,
        "n_bar": n_bar,
    }


def check_sum_rules(solver, orders=(0, 1, 2, 3)) -> dict[str, dict]:
    r"""
    Per-flavor comparison of the closed-form moments against the numerically
    integrated :math:`\int d\omega\, \omega^m A(\omega)`.

    The main entry point of this module: run this on a converged solver to
    get an independent, reference-free check of solution quality.

    Parameters
    ----------
    solver : neq_kk_ipt_solver.Solver
        A solver instance that has already converged (:meth:`Solver.solve`
        has been called successfully).
    orders : iterable of int, default=(0, 1, 2, 3)
        Which moment orders to check. The :math:`m=0` entry has no closed
        form of its own; it is reported as the normalization
        :math:`\int A = 1`, which is the cheapest way to see whether the
        frequency grid is wide enough for the higher moments to mean
        anything.

    Returns
    -------
    dict[str, dict]
        One entry per flavor, each with a ``"moments"`` sub-dict (per order:
        ``closed_form``, ``integrated``, ``abs_dev``, ``rel_dev``) plus
        ``D1``, ``D2``, ``C_bar``, ``B_tilde_bar``, ``n_bar`` from
        :func:`closed_form_moments`.
    """
    report = {}

    for fl in solver.flavors:
        exact = closed_form_moments(solver, fl)
        numeric = spectral_moments(solver.w, solver.GF[fl].R, orders=orders)

        entries = {}
        for m in orders:
            lhs = 1.0 if m == 0 else exact[f"M{m}"]
            rhs = numeric[m]
            scale = max(abs(lhs), abs(rhs), 1e-30)
            entries[m] = {
                "closed_form": lhs,
                "integrated": rhs,
                "abs_dev": abs(lhs - rhs),
                "rel_dev": abs(lhs - rhs) / scale,
            }

        report[fl] = {
            "moments": entries,
            "D1": exact["D1"],
            "D2": exact["D2"],
            "C_bar": exact["C_bar"],
            "B_tilde_bar": exact["B_tilde_bar"],
            "n_bar": exact["n_bar"],
        }

    return report
