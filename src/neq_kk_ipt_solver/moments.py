"""
Spectral-moment sum rules for the nonequilibrium Anderson impurity model,
spin-resolved.

The retarded impurity GF admits the exact large-frequency expansion

    G^R_s(w) = sum_m M_m^s / w^(m+1),   M_m^s = < { L^m f_s , f_s^dag } >,

with L X = [X, H]. The nested-commutator form is a state-independent operator
identity, so the moments hold in any steady state, in or out of equilibrium
(the expectation values are taken in the actual NESS). Writing

    D_k^s = -(1/pi) int dw w^(k-1) Im Delta^R_s(w)

for the coefficients of Delta^R_s(w) = sum_k D_k^s / w^k, the first three read

    M_1^s = eps_s + U n_sbar
    M_2^s = (M_1^s)^2 + U^2 n_sbar (1 - n_sbar) + D_1^s
    M_3^s = eps_s^3 + U n_sbar (3 eps_s^2 + 3 eps_s U + U^2)
            + 2 D_1^s M_1^s + D_2^s
            + U^2 C_sbar

M_1 and M_2 are the spin-resolved form of the sum rules already used in the
preprint; both close on the occupations alone, so they are external benchmarks.

M_3 does not. Its last term is the two-particle correlator

    C_sbar = Re < (2 n_s - 1) A^dag_sbar >,   A^dag_sbar = sum_lp t_l,sbar c^dag_lp,sbar f_sbar

i.e. the nonequilibrium Keldysh analogue of the Potthoff-Wegner-Nolting
[Phys. Rev. B 55, 16132 (1997)] m=3 band shift, and equal to
n_sbar (1 - n_sbar) (B_tilde_sbar - eps_sbar). Note the flavor index: the
sigma moment needs the OPPOSITE spin's band shift -- the same B_tilde[other]
that Solver.calc_SE already consumes.

Using Sigma_s G_s = U <<n_sbar f_s ; f_s^dag>> and Langreth on the triple
product, C_sbar is the functional of the converged solution

    C_sbar = Re { -i int dw/(2 pi) [ Q_sbar G_sbar Delta_sbar ]^< },  Q = 2 Sigma/U - 1

which is exactly what Solver._calc_band_shift_interacting integrates. The M_3
sum rule is therefore a consistency check between two different functionals of
the same (G, Sigma) -- not an absolute benchmark like M_1/M_2 -- and is a
direct probe of whether the self-energy ansatz carries the right m=3 structure.

Author: Tommaso Maria Mazzocchi
"""
import numpy as np

from .utils import Keldysh


def _lesser(X: Keldysh) -> np.ndarray:
    """X^< = X^K/2 - i Im X^R."""
    return X.K / 2.0 - 1j * np.imag(X.R)


def hybridization_moments(w: np.ndarray, Delta_R: np.ndarray, kmax: int = 2) -> dict[int, float]:
    """
    Coefficients D_k of the large-w expansion Delta^R(w) = sum_k D_k / w^k,

        D_k = -(1/pi) int dw w^(k-1) Im Delta^R(w),

    so D_1 is the total spectral weight of the hybridization (the preprint's
    D_1) and D_2 its first moment.
    """
    im_Delta = np.imag(Delta_R)
    return {
        k: float(-np.trapezoid(w ** (k - 1) * im_Delta, w) / np.pi)
        for k in range(1, kmax + 1)
    }


def spectral_moments(w: np.ndarray, GF_R: np.ndarray, orders=(0, 1, 2, 3)) -> dict[int, float]:
    """Numerically integrated moments int dw w^m A(w), with A = -Im G^R / pi."""
    A = -np.imag(GF_R) / np.pi
    return {m: float(np.trapezoid(w ** m * A, w)) for m in orders}


def band_shift_correlator(
    w: np.ndarray, GF: Keldysh, SE: Keldysh, Delta: Keldysh, U: float
) -> float:
    """
    C_s = Re < (2 n_sbar - 1) A^dag_s > = Re { -i int dw/(2 pi) [Q G Delta]^< },
    with Q = 2 Sigma/U - 1 (whose lesser component is 2 Sigma^</U, the -1 being
    instantaneous) and the Langreth rule

        [Q G Delta]^< = Q^R G^R Delta^< + Q^R G^< Delta^A + Q^< G^A Delta^A.

    Same integral as Solver._calc_band_shift_interacting, but returned raw --
    without dividing by n(1-n) -- since that is the combination the m=3 sum
    rule needs, and it stays well behaved as n approaches 0 or 1.
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
    """
    Keldysh Galitskii-Migdal double occupancy evaluated from ONE spin channel,

        n_d = -i/(2 pi U) int dw [Sigma^R(w) G^<(w) + Sigma^<(w) G^A(w)],

    the same expression Solver.calc_occupations uses, but with the flavor left
    to the caller instead of being fixed to flavors[0].

    Exactly, the result is independent of which spin is used: the equation of
    motion gives Sigma_s G_s = U <<n_sbar f_s ; f_s^dag>>, whose equal-time
    lesser component is U <n_sbar n_s> either way. Evaluating it for both spins
    and comparing is therefore a free internal-consistency check on an
    approximate solution -- one that probes the two-particle sector, unlike the
    spectral moment sum rules above. It has no content when the two flavors are
    equivalent by symmetry; it only bites for spin-dependent setups.
    """
    if abs(U) < 1e-12:
        raise ValueError("The Galitskii-Migdal double occupancy requires finite U.")

    GF_A = np.conj(GF.R)
    integrand = SE.R * _lesser(GF) + _lesser(SE) * GF_A

    return float(np.real(-1j / (2.0 * np.pi * U) * np.trapezoid(integrand, w)))


def closed_form_moments(solver, fl: str) -> dict[str, float]:
    """
    The three sum rules above, evaluated for flavor 'fl' from a converged
    solver: its onsite energies, U, self-consistent occupations, hybridization
    moments, and (for m=3) the opposite flavor's band-shift correlator.
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
    """
    Per-flavor comparison of the closed-form moments against the numerically
    integrated int dw w^m A(w).

    The m=0 entry has no closed form of its own; it is reported as the
    normalization int A = 1, which is the cheapest way to see whether the
    frequency grid is wide enough for the higher moments to mean anything.
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
