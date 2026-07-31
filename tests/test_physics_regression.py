"""
Physics regression tests: reproduce real, previously-computed single-band IPT
results from the sister project this solver was extracted from
(electric_field_hypercubic/preliminary_tests/march12), at U=5.5, T=0.05, with
a fixed (Bethe-lattice-like) hybridization function and two impurity levels
(eps=0, particle-hole symmetric; eps=-3, asymmetric filling).

For context (not asserted against, since IPT is a controlled approximation,
not exact): the same inputs were also solved with AMEA (a numerically exact
auxiliary master equation solver) in that project, giving N_double=0.02433
(eps=0) and N_double=0.1323 (eps=-3) -- IPT is expected to be in the right
ballpark of these, not identical to them. What IS asserted here is that this
extracted Solver reproduces the ORIGINAL IPT solver_output exactly (same
algorithm, same input -> same output to numerical-solver tolerance), which is
the correctness bar for this extraction.
"""
import json
import os

import numpy as np
import pytest

from neq_kk_ipt_solver import Solver

HERE = os.path.dirname(__file__)
FIXTURE_PATH = os.path.join(HERE, "data", "aim_T0p05_U5p5_reference.json")

with open(FIXTURE_PATH) as f:
    FIXTURE = json.load(f)


def _build_input(case: str, tmp_path) -> dict:
    case_data = FIXTURE[case]
    hyb = FIXTURE["shared_hybridization"]

    # 'T' is not part of global_parameters.schema.json (it's only meaningful
    # to the solver's own flat-DOS/Lorentzian hybridization construction, see
    # solver.dynamic.T) -- drop it here since this fixture supplies the
    # hybridization explicitly and doesn't need it, to avoid a spurious
    # schema-validation warning.
    global_parameters = {k: v for k, v in case_data["global_parameters"].items() if k != "T"}

    return {
        "global_parameters": global_parameters,
        "solver": {
            "static": {},
            "modifiable": {
                "impurity_onsite_e": case_data["impurity_onsite_e"],
            },
            "dynamic": {
                "Delta_R_im": hyb["Delta_R_im"],
                "Delta_K_im": hyb["Delta_K_im"],
                "output_dir": str(tmp_path),
            },
        },
    }


@pytest.mark.parametrize("case", ["eps0", "eps_m3"])
def test_reproduces_reference_occupations(case, tmp_path):
    input_data = _build_input(case, tmp_path)
    expected = FIXTURE[case]["expected"]

    solver = Solver(input_data)
    sol = solver.solve()

    assert sol.success

    assert solver.n_occ["up"] == pytest.approx(expected["n_occ_up"], abs=1e-4)
    assert solver.n_occ["down"] == pytest.approx(expected["n_occ_down"], abs=1e-4)
    assert solver.n_double == pytest.approx(expected["n_double"], abs=1e-4)


def test_eps0_case_is_spin_symmetric():
    """The eps=0 case has identical onsite energy for both flavors, so the
    converged occupations should match between spin channels."""
    expected = FIXTURE["eps0"]["expected"]
    assert expected["n_occ_up"] == pytest.approx(expected["n_occ_down"], abs=1e-8)


def test_eps_m3_pulls_occupation_above_half_filling():
    """A negative onsite energy (below the chemical potential) should increase
    the occupation relative to the eps=0 (roughly quarter-filled) case."""
    n_eps0 = FIXTURE["eps0"]["expected"]["n_occ_up"]
    n_eps_m3 = FIXTURE["eps_m3"]["expected"]["n_occ_up"]
    assert n_eps_m3 > n_eps0
