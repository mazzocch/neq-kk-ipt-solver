"""
Nonequilibrium physics regression tests: reproduce real, previously-computed
single-band IPT results from electric_field_hypercubic/runs_kkipt_new, a
genuine voltage-biased (V=1.0) steady state -- U=4, T=0.1175, flat-DOS lead
hybridization (Gamma set by t_l/t_r/D_l/D_r), two impurity levels (eps=0,
particle-hole symmetric; eps=-2.25, asymmetric filling).

This exercises the solver's actual nonequilibrium capability (a real chemical
potential bias between the two leads, mu_l = mu+V/2 != mu_r = mu-V/2, hence a
genuinely non-thermal Delta^K), not just the equilibrium (V=0) case already
covered in test_physics_regression.py.

As in that file: this asserts exact reproduction of the ORIGINAL IPT
solver_output (same algorithm, same input -> same output), not agreement
with any numerically-exact reference -- IPT is a controlled approximation.
"""
import json
import os

import pytest

from neq_kk_ipt_solver import Solver

HERE = os.path.dirname(__file__)
FIXTURE_PATH = os.path.join(HERE, "data", "aim_U4_T0p1175_V1_nonequilibrium_reference.json")

with open(FIXTURE_PATH) as f:
    FIXTURE = json.load(f)


def _build_input(case: str, tmp_path) -> dict:
    case_data = FIXTURE[case]
    hyb = case_data["hybridization"]

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


@pytest.mark.parametrize("case", ["eps0", "eps_m2p25"])
def test_reproduces_nonequilibrium_reference_occupations(case, tmp_path):
    input_data = _build_input(case, tmp_path)
    expected = FIXTURE[case]["expected"]

    solver = Solver(input_data)
    sol = solver.solve()

    assert sol.success

    assert solver.n_occ["up"] == pytest.approx(expected["n_occ_up"], abs=1e-4)
    assert solver.n_occ["down"] == pytest.approx(expected["n_occ_down"], abs=1e-4)
    assert solver.n_double == pytest.approx(expected["n_double"], abs=1e-4)


def test_eps0_nonequilibrium_case_is_spin_symmetric():
    expected = FIXTURE["eps0"]["expected"]
    assert expected["n_occ_up"] == pytest.approx(expected["n_occ_down"], abs=1e-8)


def test_eps_m2p25_pulls_occupation_above_eps0_case():
    """A negative onsite energy (below the chemical potential) should increase
    the occupation relative to the eps=0 case, at the same bias voltage."""
    n_eps0 = FIXTURE["eps0"]["expected"]["n_occ_up"]
    n_eps_m2p25 = FIXTURE["eps_m2p25"]["expected"]["n_occ_up"]
    assert n_eps_m2p25 > n_eps0
