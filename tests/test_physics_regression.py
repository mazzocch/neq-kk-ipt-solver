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
the correctness bar for this extraction. This includes the full Green's
function and self-energy (not just the scalar occupations), so a change that
gets the observables right by accident but the spectral data wrong would
still be caught.

n_double in the fixture is NOT the "N_double" field originally stored in
those files -- it is recomputed from the raw GF/SE arrays via the exact
Keldysh Galitskii-Migdal expression (see scripts/build_reference_fixtures.py),
since a very early version of this solver used an approximate formula that
coincides with the exact one only in equilibrium (for these equilibrium
cases the two agree to ~1e-14/1e-16 anyway; the distinction only matters for
the nonequilibrium fixture, see test_physics_regression_nonequilibrium.py).
"""
import json
import os

import numpy as np
import pytest

from neq_kk_ipt_solver import Solver

HERE = os.path.dirname(__file__)
FIXTURE_JSON_PATH = os.path.join(HERE, "data", "aim_T0p05_U5p5_reference.json")
FIXTURE_NPZ_PATH = os.path.join(HERE, "data", "aim_T0p05_U5p5_reference.npz")

with open(FIXTURE_JSON_PATH) as f:
    FIXTURE = json.load(f)

ARRAYS = np.load(FIXTURE_NPZ_PATH)


def _build_input(case: str, tmp_path) -> dict:
    case_data = FIXTURE[case]

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
                "Delta_R_im": {
                    "up": ARRAYS["hyb_Delta_R_im_up"].tolist(),
                    "down": ARRAYS["hyb_Delta_R_im_down"].tolist(),
                },
                "Delta_K_im": {
                    "up": ARRAYS["hyb_Delta_K_im_up"].tolist(),
                    "down": ARRAYS["hyb_Delta_K_im_down"].tolist(),
                },
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


@pytest.mark.parametrize("case", ["eps0", "eps_m3"])
@pytest.mark.parametrize("fl", ["up", "down"])
def test_reproduces_reference_greens_function_and_self_energy(case, fl, tmp_path):
    input_data = _build_input(case, tmp_path)
    prefix = f"{case}_{fl}"

    solver = Solver(input_data)
    solver.solve()

    ref_GF_R = ARRAYS[f"{prefix}_GF_R_re"] + 1j * ARRAYS[f"{prefix}_GF_R_im"]
    ref_GF_K = ARRAYS[f"{prefix}_GF_K_re"] + 1j * ARRAYS[f"{prefix}_GF_K_im"]
    ref_SE_R = ARRAYS[f"{prefix}_SE_R_re"] + 1j * ARRAYS[f"{prefix}_SE_R_im"]
    ref_SE_K = ARRAYS[f"{prefix}_SE_K_re"] + 1j * ARRAYS[f"{prefix}_SE_K_im"]

    np.testing.assert_allclose(solver.GF[fl].R, ref_GF_R, atol=1e-6)
    np.testing.assert_allclose(solver.GF[fl].K, ref_GF_K, atol=1e-6)
    np.testing.assert_allclose(solver.SE[fl].R, ref_SE_R, atol=1e-6)
    np.testing.assert_allclose(solver.SE[fl].K, ref_SE_K, atol=1e-6)


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
