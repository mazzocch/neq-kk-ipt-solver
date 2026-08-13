"""
Checks the spin-resolved m=1,2,3 spectral moment sum rules (see
neq_kk_ipt_solver.moments) against the numerically integrated
int dw w^m A_s(w), for the voltage-biased steady state V=1.

Three parts:

  A. Grid convergence. The m=3 integrand carries a w^3 weight, so it is far
     more sensitive to the frequency window than m=1/2: the IPT diagram's
     support is roughly three times the hybridization bandwidth (the
     self-energy is a double convolution of the Weiss field), so a grid that
     is perfectly adequate for the lower moments can silently truncate this
     one. Part A sweeps w_max at fixed grid spacing, with the plain
     KK-IPT-n0 scheme, to find where int dw w^3 A stops moving.

  B. The actual comparison the sum rule is for: plain KK-IPT-n0 versus the
     Potthoff-Wegner-Nolting m=3 band-shift correction, on the converged grid
     from part A, for the two fillings of the preprint (eps = 0 and the
     near-half-filling eps = -2.25).

  C. The same comparison on the committed V=1 regression fixture, which pins
     the exact hybridization arrays of a previously published run. Its grid
     (w_max=30, N_points=10001) is narrower than the recommended w_max=50 but
     finer; part A's verdict applies here too and is reported alongside.

  D. A spin-dependent hybridization with off-centre lead bands, which is the
     only setup here that exercises the D_2 term and the sigma/sigma-bar index
     structure of the m=3 correlator.

Neither n_double nor the occupations respond to the grid at these parameters:
they are w^0-weighted integrals whose integrands are already negligible in the
tails. The moment sum rules are w^m-weighted, so the same small tail defect is
amplified by w^m, which makes them a considerably more sensitive probe of grid
adequacy than n_double is.

Parameters follow the preprint: U = 4 Gamma, T = 0.1175 Gamma, D = 10 Gamma,
t_l = t_r = 1/sqrt(2), T_fict = 1/2, mu = 0, V = 1 Gamma.

Not part of the installed package; a diagnostic driver.
"""
import json
import os

import numpy as np

from neq_kk_ipt_solver import Solver
from neq_kk_ipt_solver.moments import check_sum_rules
from neq_kk_ipt_solver.utils import fermi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "tests", "data")
FIXTURE_STEM = os.path.join(DATA_DIR, "aim_U4_T0p1175_V1_nonequilibrium_reference")

U = 4.0
T = 0.1175
V = 1.0
D_HALF_BANDWIDTH = 10.0
T_FICT = 0.5
HOPPING = 1.0 / np.sqrt(2.0)

ONSITE_CASES = {"eps0": 0.0, "eps_m2p25": -2.25}

# Spacing of the recommended production grid (w_max=50, N_points=10001), held
# fixed across the w_max sweep so the sweep isolates window width from
# resolution. Both observables below are insensitive to it: sweeping dw from
# 0.02 to 0.0025 leaves n_double and every moment deviation unchanged to the
# digits printed. It is the WINDOW, not the spacing, that the moments respond
# to -- and only weakly (see part A).
DW = 2 * 50.0 / (10001 - 1)


def _n_points(w_max: float, dw: float = DW) -> int:
    n = int(round(2 * w_max / dw)) + 1
    return n if n % 2 == 1 else n + 1


def flat_dos_input(eps: float, w_max: float, potthoff: bool, tmp_dir: str) -> dict:
    """Solver input using the built-in flat-DOS reservoir at the preprint's parameters."""
    return {
        "global_parameters": {
            "N_points": _n_points(w_max),
            "w_max": w_max,
            "U": U,
            "flavors": ["up", "down"],
        },
        "solver": {
            "static": {"store": False, "use_potthoff_band_shift": potthoff,
                   # diagnostic script: report non-convergence, do not raise
                   "on_convergence_failure": "warn"},
            "modifiable": {"impurity_onsite_e": {"up": eps, "down": eps}},
            "dynamic": {
                "T": T,
                "T_fict": T_FICT,
                "D_l": D_HALF_BANDWIDTH,
                "D_r": D_HALF_BANDWIDTH,
                "t_l": HOPPING,
                "t_r": HOPPING,
                "mu": 0.0,
                "V": V,
                "output_dir": tmp_dir,
            },
        },
    }


def fixture_input(case: str, potthoff: bool, tmp_dir: str) -> dict:
    """Solver input reusing the committed fixture's exact hybridization arrays."""
    with open(f"{FIXTURE_STEM}.json") as f:
        meta = json.load(f)
    arrays = np.load(f"{FIXTURE_STEM}.npz")

    global_parameters = {
        k: v for k, v in meta[case]["global_parameters"].items() if k != "T"
    }

    return {
        "global_parameters": global_parameters,
        "solver": {
            "static": {"store": False, "use_potthoff_band_shift": potthoff,
                   # diagnostic script: report non-convergence, do not raise
                   "on_convergence_failure": "warn"},
            "modifiable": {"impurity_onsite_e": meta[case]["impurity_onsite_e"]},
            "dynamic": {
                "Delta_R_im": {
                    fl: arrays[f"{case}_hyb_Delta_R_im_{fl}"].tolist()
                    for fl in ("up", "down")
                },
                "Delta_K_im": {
                    fl: arrays[f"{case}_hyb_Delta_K_im_{fl}"].tolist()
                    for fl in ("up", "down")
                },
                "output_dir": tmp_dir,
            },
        },
    }


def spin_asymmetric_input(w_max: float, potthoff: bool, tmp_dir: str) -> dict:
    """
    Explicit per-flavor hybridization arrays: spin-dependent hoppings and
    off-centre lead bands. This is the case the preprint does not cover --
    it makes n_up != n_down, eps_up != eps_down and, crucially, D_2 != 0,
    so it exercises the two parts of the m=3 sum rule that the paramagnetic
    flat-DOS setups leave completely untested: the D_2 term and the sigma /
    sigma-bar index structure of the band-shift correlator.

    Deliberately still a finite band with exponentially smeared edges. A
    Lorentzian bath would not do: its Im Delta^R ~ 1/w^2 tail makes D_2 only
    conditionally convergent and int dw w^3 A(w) logarithmically divergent,
    so the m=3 moment does not exist at all.
    """
    n_points = _n_points(w_max)
    w = np.linspace(-w_max, w_max, n_points)

    mu_l, mu_r = V / 2, -V / 2

    # (hopping, band centre) per lead, per flavor
    leads = {
        "up": {"l": (0.8, 1.5), "r": (0.6, -2.0)},
        "down": {"l": (0.5, 0.5), "r": (0.9, -1.0)},
    }
    onsite = {"up": -1.2, "down": -2.6}

    Delta_R_im, Delta_K_im = {}, {}
    for fl, cfg in leads.items():
        dR = np.zeros_like(w)
        dK = np.zeros_like(w)
        for lead, mu_lead in (("l", mu_l), ("r", mu_r)):
            t, centre = cfg[lead]
            band = -(
                (1 - fermi(w, centre - D_HALF_BANDWIDTH, T_FICT))
                * fermi(w, centre + D_HALF_BANDWIDTH, T_FICT)
            )
            dR += t ** 2 * band
            dK += 2 * t ** 2 * band * (1 - 2 * fermi(w, mu_lead, T))
        Delta_R_im[fl] = dR.tolist()
        Delta_K_im[fl] = dK.tolist()

    return {
        "global_parameters": {
            "N_points": n_points,
            "w_max": w_max,
            "U": U,
            "flavors": ["up", "down"],
        },
        "solver": {
            "static": {
                "store": False,
                "spin_sym": False,
                "ph_sym": False,
                "use_potthoff_band_shift": potthoff,
                # diagnostic script: report non-convergence, do not raise
                "on_convergence_failure": "warn",
            },
            "modifiable": {"impurity_onsite_e": onsite},
            "dynamic": {
                "Delta_R_im": Delta_R_im,
                "Delta_K_im": Delta_K_im,
                "output_dir": tmp_dir,
            },
        },
    }


def analytic_hybridization_moments() -> dict[str, tuple[float, float]]:
    """
    D_1, D_2 for spin_asymmetric_input's bands, in the sharp-edge limit:
    a lead of weight t^2 and half-width D centred at c contributes
    2 D t^2 / pi to D_1 and 2 D c t^2 / pi to D_2.
    """
    leads = {
        "up": {"l": (0.8, 1.5), "r": (0.6, -2.0)},
        "down": {"l": (0.5, 0.5), "r": (0.9, -1.0)},
    }
    scale = 2 * D_HALF_BANDWIDTH / np.pi
    return {
        fl: (
            scale * sum(t ** 2 for t, _ in cfg.values()),
            scale * sum(t ** 2 * c for t, c in cfg.values()),
        )
        for fl, cfg in leads.items()
    }


def run(input_data: dict) -> tuple[Solver, dict]:
    solver = Solver(input_data)
    sol = solver.solve()
    if not sol.success:
        print(f"  !! root solve did NOT converge: {sol.message}")
    return solver, check_sum_rules(solver)


def moment_row(label: str, report: dict, fl: str = "up") -> str:
    m = report[fl]["moments"]
    return (
        f"{label:<26}"
        + "".join(
            f"{m[k]['closed_form']:>12.5f}{m[k]['integrated']:>12.5f}{m[k]['rel_dev']:>11.2e}"
            for k in (1, 2, 3)
        )
    )


HEADER = (
    f"{'':<26}"
    + "".join(f"{'M' + str(k) + ' exact':>12}{'integral':>12}{'rel.dev':>11}" for k in (1, 2, 3))
)


def part_a(tmp_dir: str) -> None:
    print("\n" + "=" * 100)
    print("A. GRID CONVERGENCE of int dw w^m A(w)   [flat DOS, plain KK-IPT-n0, V=1]")
    print("   Fixed spacing dw = %.4f; the IPT diagram's support reaches ~3 x D = %.0f." % (DW, 3 * D_HALF_BANDWIDTH))
    print("=" * 100)

    for case, eps in ONSITE_CASES.items():
        print(f"\n  eps = {eps}   ({case})")
        print("  " + HEADER)
        for w_max in (30.0, 40.0, 50.0, 60.0):
            _, report = run(flat_dos_input(eps, w_max, potthoff=False, tmp_dir=tmp_dir))
            label = f"w_max={w_max:.0f}, N={_n_points(w_max)}"
            print("  " + moment_row(label, report))


def part_b(w_max: float, tmp_dir: str) -> None:
    print("\n" + "=" * 100)
    print(f"B. PLAIN vs POTTHOFF  [flat DOS, V=1, w_max={w_max:.0f}, N={_n_points(w_max)}]")
    print("=" * 100)

    for case, eps in ONSITE_CASES.items():
        print(f"\n  eps = {eps}   ({case})")
        print("  " + HEADER)
        for potthoff in (False, True):
            solver, report = run(flat_dos_input(eps, w_max, potthoff, tmp_dir))
            label = "Potthoff m=3" if potthoff else "plain KK-IPT-n0"
            print("  " + moment_row(label, report))
            summarize(solver, report, potthoff)


def part_c(tmp_dir: str) -> None:
    print("\n" + "=" * 100)
    print("C. PLAIN vs POTTHOFF  [committed V=1 regression fixture, its own grid: w_max=30]")
    print("=" * 100)

    for case in ONSITE_CASES:
        print(f"\n  {case}")
        print("  " + HEADER)
        for potthoff in (False, True):
            solver, report = run(fixture_input(case, potthoff, tmp_dir))
            label = "Potthoff m=3" if potthoff else "plain KK-IPT-n0"
            print("  " + moment_row(label, report))
            summarize(solver, report, potthoff)


def summarize(solver: Solver, report: dict, potthoff: bool) -> None:
    up = report["up"]
    line = (
        f"      n_up={solver.n_occ['up']:.6f}  n_d={solver.n_double:.6f}  "
        f"D1={up['D1']:.4f}  D2={up['D2']:+.2e}  "
        f"C_bar={up['C_bar']:+.6f}  B_tilde_bar={up['B_tilde_bar']:+.6f}"
    )
    if potthoff:
        line += (
            f"\n      band-shift inner: err={solver.band_shift_error:.2e}, "
            f"iters={solver.band_shift_iterations}, "
            f"diverged trial states={solver.band_shift_diverged_count}"
        )
    print(line)


def part_d(w_max: float, tmp_dir: str) -> None:
    print("\n" + "=" * 100)
    print(f"D. SPIN-DEPENDENT hybridization + onsite energies, D_2 != 0  [V=1, w_max={w_max:.0f}]")
    print("   Tests the two ingredients the paramagnetic setups cannot: the D_2 term")
    print("   and the sigma/sigma-bar index structure of the band-shift correlator.")
    print("=" * 100)

    analytic = analytic_hybridization_moments()

    for potthoff in (False, True):
        solver, report = run(spin_asymmetric_input(w_max, potthoff, tmp_dir))
        label = "Potthoff m=3" if potthoff else "plain KK-IPT-n0"
        print(f"\n  {label}")
        print("  " + HEADER)
        for fl in solver.flavors:
            print("  " + moment_row(f"  flavor {fl}", report, fl=fl))
            d = report[fl]
            D1_ref, D2_ref = analytic[fl]
            print(
                f"      n_{fl}={solver.n_occ[fl]:.6f}  "
                f"D1={d['D1']:.4f} (sharp-edge {D1_ref:.4f})  "
                f"D2={d['D2']:+.4f} (sharp-edge {D2_ref:+.4f})  "
                f"C_bar={d['C_bar']:+.6f}"
            )
        print(f"      n_d={solver.n_double:.6f}")
        if potthoff:
            print(
                f"      band-shift inner: err={solver.band_shift_error:.2e}, "
                f"iters={solver.band_shift_iterations}, "
                f"diverged trial states={solver.band_shift_diverged_count}"
            )


def main() -> None:
    tmp_dir = os.environ.get("MOMENT_CHECK_OUTPUT_DIR", "/tmp/moment_check_out")
    part_a(tmp_dir)
    part_b(w_max=50.0, tmp_dir=tmp_dir)
    part_c(tmp_dir)
    part_d(w_max=50.0, tmp_dir=tmp_dir)


if __name__ == "__main__":
    main()
