"""
Spectral-moment sum rule regression tests (see neq_kk_ipt_solver.moments).

Pins two things:

  * the m=1 and m=2 sum rules, which close on the occupations alone and are
    therefore genuine external benchmarks, hold to well under 0.1% for both
    self-energy schemes;

  * the m=3 sum rule, whose closed form additionally needs the
    Potthoff-Wegner-Nolting band-shift correlator, is violated by a few
    percent by plain KK-IPT-n0 and satisfied to the numerical floor once the
    m=3 band-shift correction is switched on.

The second is the point of the correction, and it is a large, unambiguous
effect (two to three orders of magnitude), so the assertions below are loose
enough not to be brittle while still failing if the correction stops working.

Everything runs on the grid of the committed reference fixtures -- w_max=30,
N_points=10001, which is what the original production runs used -- at the
representative bias V=1. Both n_double and the moment deviations were checked
to be insensitive to that choice (see scripts/check_spectral_moments.py).

Only spin-symmetric hybridizations are covered: the spin-dependent case has no
published reference yet. The closed form itself is spin-resolved and is
exercised per flavor.
"""
import json
import os

import numpy as np
import pytest

from neq_kk_ipt_solver import Solver
from neq_kk_ipt_solver.moments import (
    band_shift_correlator,
    check_sum_rules,
    hybridization_moments,
)

HERE = os.path.dirname(__file__)
FIXTURE_STEM = os.path.join(HERE, "data", "aim_U4_T0p1175_V1_nonequilibrium_reference")

U = 4.0
T = 0.1175
V = 1.0
N_POINTS = 10001
W_MAX = 30.0

# eps = -U/2 is the particle-hole symmetric point. There A(w) is symmetric, so
# every ODD moment vanishes identically -- M_1 = eps + U n = 0 and, term by
# term, M_3 = 0 as well. Relative deviations are meaningless there (0/0), so
# those setups get their own exactness test instead of the ratio assertions.
PH_SYMMETRIC_EPS = -U / 2


def _global_parameters() -> dict:
    return {
        "N_points": N_POINTS,
        "w_max": W_MAX,
        "U": U,
        "flavors": ["up", "down"],
    }


def _wrap(dynamic: dict, eps: float, potthoff: bool, tmp_dir: str) -> dict:
    dynamic = dict(dynamic)
    dynamic["output_dir"] = str(tmp_dir)
    return {
        "global_parameters": _global_parameters(),
        "solver": {
            "static": {
                "store": False,
                "ph_sym": False,
                "use_potthoff_band_shift": potthoff,
            },
            "modifiable": {"impurity_onsite_e": {"up": eps, "down": eps}},
            "dynamic": dynamic,
        },
    }


def flat_dos(eps: float, potthoff: bool, tmp_dir: str) -> dict:
    """The preprint's leads: D = 10 Gamma, t_l = t_r = 1/sqrt(2), T_fict = 1/2."""
    return _wrap(
        {
            "T": T,
            "T_fict": 0.5,
            "D_l": 10.0,
            "D_r": 10.0,
            "t_l": 1.0 / np.sqrt(2.0),
            "t_r": 1.0 / np.sqrt(2.0),
            "mu": 0.0,
            "V": V,
        },
        eps,
        potthoff,
        tmp_dir,
    )


def semicircular(eps: float, potthoff: bool, tmp_dir: str) -> dict:
    """
    Semi-elliptic bath, half-bandwidth 10.

    The width matters: a semicircle has a HARD band edge, so outside it
    Im Delta^R is exactly zero and any impurity feature landing there becomes a
    genuine bound state -- a delta function the frequency grid cannot represent,
    which silently costs spectral weight and breaks every sum rule including the
    m=0 normalization. With half-bandwidth 10 both Hubbard bands stay inside the
    bath band and the problem does not arise; at half-bandwidth 2 the upper
    Hubbard band sits near w = +5.9, outside it, and int A drops to ~0.97 on this
    grid. The smeared flat DOS above is immune because its Im Delta^R never
    vanishes exactly.
    """
    return _wrap(
        {"T": T, "Delta_D": 10.0, "Delta_center": 0.0, "mu": 0.0, "V": V},
        eps,
        potthoff,
        tmp_dir,
    )


def fixture_delta(eps: float, potthoff: bool, tmp_dir: str) -> dict:
    """The exact hybridization arrays of the committed V=1 production run."""
    arrays = np.load(f"{FIXTURE_STEM}.npz")
    case = "eps0" if eps == 0.0 else "eps_m2p25"
    return _wrap(
        {
            "Delta_R_im": {
                fl: arrays[f"{case}_hyb_Delta_R_im_{fl}"].tolist() for fl in ("up", "down")
            },
            "Delta_K_im": {
                fl: arrays[f"{case}_hyb_Delta_K_im_{fl}"].tolist() for fl in ("up", "down")
            },
        },
        eps,
        potthoff,
        tmp_dir,
    )


BUILDERS = {
    "flat_dos": flat_dos,
    "semicircular": semicircular,
    "fixture_delta": fixture_delta,
}

# eps = -2.25 is the preprint's near-half-filling level; eps = 0 is well away
# from it. Neither coincides with the particle-hole symmetric point -U/2 = -2,
# so M_3 is nonzero for both and the ratio assertions are meaningful.
CASES = [(hyb, eps) for hyb in BUILDERS for eps in (0.0, -2.25)]


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """Solve each (hybridization, eps, scheme) combination at most once."""
    tmp_dir = tmp_path_factory.mktemp("moment_sum_rules")
    cache: dict = {}

    def get(hyb: str, eps: float, potthoff: bool):
        key = (hyb, eps, potthoff)
        if key not in cache:
            solver = Solver(BUILDERS[hyb](eps, potthoff, tmp_dir))
            sol = solver.solve()
            cache[key] = (solver, sol, check_sum_rules(solver))
        return cache[key]

    return get


# --------------------------------------------------------------------------
# The sum rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hyb, eps", CASES)
@pytest.mark.parametrize("potthoff", [False, True], ids=["plain", "potthoff"])
def test_low_moment_sum_rules_hold(hyb, eps, potthoff, solved):
    """
    m=1 and m=2 close on the occupations alone, so both schemes must satisfy
    them -- and the m=3 correction must not spoil them. The observed deviations
    are 2e-5 to 1e-4, set by the finite frequency window.
    """
    solver, sol, report = solved(hyb, eps, potthoff)
    assert sol.success

    for fl in solver.flavors:
        moments = report[fl]["moments"]

        # Normalization first: if spectral weight is being lost off the grid,
        # every other deviation below is meaningless.
        assert moments[0]["integrated"] == pytest.approx(1.0, abs=1e-3)

        for m in (1, 2):
            entry = moments[m]
            if abs(entry["closed_form"]) < 1e-2:
                continue  # degenerate: relative deviation is 0/0
            assert entry["rel_dev"] < 5e-4


@pytest.mark.parametrize("hyb, eps", CASES)
def test_potthoff_band_shift_fixes_the_third_moment(hyb, eps, solved):
    """
    The headline check. Plain KK-IPT-n0 misses the m=3 moment by a few percent;
    the Potthoff-Wegner-Nolting band shift brings it down to the same numerical
    floor as m=1/m=2. Measured improvement is a factor 130-270.
    """
    _, plain_sol, plain = solved(hyb, eps, False)
    solver, potthoff_sol, potthoff = solved(hyb, eps, True)
    assert plain_sol.success and potthoff_sol.success

    assert solver.band_shift_diverged_count == 0
    assert solver.band_shift_error < 1e-6

    for fl in solver.flavors:
        plain_dev = plain[fl]["moments"][3]["rel_dev"]
        potthoff_dev = potthoff[fl]["moments"][3]["rel_dev"]

        assert plain_dev > 5e-3, "plain KK-IPT-n0 should visibly miss the m=3 moment"
        assert potthoff_dev < 1e-3, "the band shift should restore the m=3 moment"
        assert plain_dev / potthoff_dev > 20.0


@pytest.mark.parametrize("potthoff", [False, True], ids=["plain", "potthoff"])
def test_third_moment_vanishes_at_particle_hole_symmetry(potthoff, tmp_path):
    """
    At eps = -U/2 the spectral function is symmetric, so all odd moments are
    zero. Every term of the closed form vanishes separately: M_1 = 0, the
    atomic part cancels, D_2 = 0 for a symmetric bath, and the band-shift
    correlator vanishes by symmetry.
    """
    solver = Solver(flat_dos(PH_SYMMETRIC_EPS, potthoff, tmp_path))
    sol = solver.solve()
    assert sol.success

    report = check_sum_rules(solver)
    for fl in solver.flavors:
        moments = report[fl]["moments"]
        n = float(np.real(solver.n_occ[fl]))
        off_half_filling = abs(n - 0.5)

        # The odd moments vanish only as exactly as the filling reaches 1/2, and
        # the root solve lands there to ~1e-5 rather than to machine precision.
        # At eps = -U/2 the leading sensitivities are dM_1/dn = U and
        # dM_3/dn = U^3/4 + 2 D_1 U (~67 here), so scale the tolerances by those
        # rather than picking a number: M_3 still has to come out ~1000x smaller
        # than the -5.46 it takes at the neighbouring eps = -2.25.
        m1_slack = 10.0 * U * off_half_filling + 1e-6
        m3_slack = 5.0 * (U ** 3 / 4 + 2 * report[fl]["D1"] * U) * off_half_filling + 1e-4

        assert moments[1]["closed_form"] == pytest.approx(0.0, abs=m1_slack)
        assert moments[3]["closed_form"] == pytest.approx(0.0, abs=m3_slack)
        assert moments[3]["integrated"] == pytest.approx(0.0, abs=m3_slack)
        assert report[fl]["C_bar"] == pytest.approx(0.0, abs=1e-3)


def test_band_shift_correlator_matches_solver_internal(solved):
    """
    moments.band_shift_correlator and Solver._calc_band_shift_interacting must
    integrate the same thing -- the former returns it raw, the latter divides by
    n(1-n) and adds the onsite energy. Guards the two against drifting apart.
    """
    solver, _, _ = solved("flat_dos", -2.25, True)

    for fl in solver.flavors:
        n_fl = float(np.real(solver.n_occ[fl]))
        raw = band_shift_correlator(
            solver.w, solver.GF[fl], solver.SE[fl], solver.Delta[fl], solver.U
        )
        from_solver = solver._calc_band_shift_interacting(fl, n_fl=n_fl)
        expected = solver.impurity_onsite_e[fl] + raw / (n_fl * (1.0 - n_fl) + 1e-12)
        assert from_solver == pytest.approx(expected, rel=1e-12)


def test_third_moment_needs_the_band_shift_term(solved):
    """
    Dropping the U^2 C_sbar term from the closed form must make the m=3 sum rule
    clearly worse for the corrected solution -- i.e. the term is doing real work
    and is not an incidental small correction.
    """
    solver, _, report = solved("flat_dos", 0.0, True)

    for fl in solver.flavors:
        entry = report[fl]["moments"][3]
        without = entry["closed_form"] - solver.U ** 2 * report[fl]["C_bar"]
        dev_without = abs(without - entry["integrated"]) / abs(entry["integrated"])
        assert dev_without > 50 * entry["rel_dev"]
