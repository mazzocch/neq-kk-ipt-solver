"""
Physics regression tests for a SPIN-DEPENDENT hybridization in equilibrium:
a Lorentzian centred at +1 for spin up and at -1 for spin down, gamma = 1, at
T = 0.02 and T = 0.05, with the onsite energies tracking -U/2.

The two existing regression files both use spin-symmetric hybridizations, so
nothing else in the suite exercises the spin-resolved machinery: the
sigma/sigma-bar index structure of the self-energy, the per-flavor
hybridization, or the full four-dimensional root solve. The U values straddle
the point at which the majority spin changes over, which is where the solution
is most sensitive to any change in the ansatz.

As in the other regression files, this asserts exact reproduction of previously
computed output -- same algorithm, same input, same output -- and not agreement
with a numerically exact reference: IPT is a controlled approximation, and what
is pinned here is what this scheme converges to, not what the true solution is.
The fixture is regenerated, deliberately, by
scripts/build_spin_dependent_fixture.py.
"""
import json
import os

import numpy as np
import pytest

from neq_kk_ipt_solver import Solver

HERE = os.path.dirname(__file__)
FIXTURE_PATH = os.path.join(HERE, "data", "spin_dependent_lorentzian_reference.json")

with open(FIXTURE_PATH) as f:
    FIXTURE = json.load(f)

SETUP = FIXTURE["setup"]
CASES = sorted(FIXTURE["cases"])
FLAVORS = ["up", "down"]


def build_input(case: dict, tmp_path) -> dict:
    U = case["U"]
    return {
        "global_parameters": {
            "N_points": SETUP["N_points"], "w_max": SETUP["w_max"],
            "U": U, "flavors": list(FLAVORS),
        },
        "solver": {
            "static": {
                "store": False, "spin_sym": False, "ph_sym": False,
                "use_potthoff_band_shift": case["scheme"] == "potthoff",
            },
            "modifiable": {"impurity_onsite_e": {fl: -U / 2 for fl in FLAVORS}},
            "dynamic": {
                "T": case["T"],
                "Delta_center": dict(SETUP["Delta_center"]),
                "Delta_gamma": {fl: SETUP["Delta_gamma"] for fl in FLAVORS},
                "mu": SETUP["mu"], "V": SETUP["V"], "output_dir": str(tmp_path),
            },
        },
    }


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("spin_dependent_regression")
    cache: dict = {}

    def get(key: str):
        if key not in cache:
            case = FIXTURE["cases"][key]
            solver = Solver(build_input(case, tmp_dir))
            cache[key] = (solver, solver.solve(), case)
        return cache[key]

    return get


@pytest.mark.parametrize("key", CASES)
def test_reproduces_the_reference_occupations(key, solved):
    solver, sol, case = solved(key)

    assert sol.success
    # sol.success now means the residual really is small, not merely that the
    # optimizer was content: Levenberg-Marquardt can report success far from a
    # root (see tests/test_convergence_handling.py), so assert it directly. The
    # pinned values are only meaningful for a genuinely self-consistent solution.
    assert solver.solve_residual < 1e-9

    assert float(np.real(solver.n_occ["up"])) == pytest.approx(case["n_occ_up"], abs=1e-6)
    assert float(np.real(solver.n_occ["down"])) == pytest.approx(case["n_occ_down"], abs=1e-6)
    assert float(solver.n_double) == pytest.approx(case["n_double"], abs=1e-6)


@pytest.mark.parametrize("key", CASES)
def test_particle_hole_spin_flip_symmetry_holds(key, solved):
    """
    The onsite energy sits at -U/2 and the two Lorentzians are mirror images, so
    the model is invariant under the combined particle-hole x spin-flip
    operation and every solution must satisfy n_up + n_down = 1. Exact, and
    independent of the fixture.
    """
    solver, _, _ = solved(key)
    total = float(np.real(solver.n_occ["up"]) + np.real(solver.n_occ["down"]))
    assert total == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("T", [0.02, 0.05])
def test_majority_spin_changes_over_with_coupling(T, solved):
    """
    Both schemes give an up-majority solution at U=4 and a down-majority one at
    U=8, at both temperatures. The solution is unique at each U, so this is a
    property of the KK-IPT self-consistency itself and not a branch the solver
    happened to land on; pinning it guards a qualitative feature that a change
    to the ansatz would be likely to move.
    """
    tag = f"T{T:g}".replace(".", "p")
    for scheme in ("plain", "potthoff"):
        low, _, _ = solved(f"{scheme}_{tag}_U4")
        high, _, _ = solved(f"{scheme}_{tag}_U8")
        assert float(np.real(low.n_occ["up"])) > 0.5
        assert float(np.real(high.n_occ["up"])) < 0.5


@pytest.mark.parametrize("T", [0.02, 0.05])
def test_the_two_schemes_actually_differ(T, solved):
    """
    Guards against the Potthoff band shift silently doing nothing, which is
    exactly what a mishandled pre-solve would look like from the outside.
    """
    tag = f"T{T:g}".replace(".", "p")
    plain, _, _ = solved(f"plain_{tag}_U8")
    potthoff, _, _ = solved(f"potthoff_{tag}_U8")
    assert abs(float(np.real(plain.n_occ["up"])) -
               float(np.real(potthoff.n_occ["up"]))) > 1e-3
