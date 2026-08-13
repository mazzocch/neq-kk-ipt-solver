"""
Tests for the semi-elliptic ("semicircular") hybridization option.

Covers the analytic moments of the band itself, per-flavor (spin-dependent)
parameters, the mutually exclusive shape parameters, and the optional constant
broadening Delta_eta that removes the hard band edges. The bound state that
motivates Delta_eta is a real hazard: a semicircle has Im Delta^R identically
zero outside the band, so any impurity feature landing there becomes a delta
function the frequency grid cannot represent.
"""
import numpy as np
import pytest

from neq_kk_ipt_solver import Solver
from neq_kk_ipt_solver.moments import check_sum_rules, hybridization_moments

U = 4.0
T = 0.1175
V = 1.0
N_POINTS = 10001
W_MAX = 30.0


def _wrap(dynamic: dict, eps: float, potthoff: bool, tmp_dir) -> dict:
    dynamic = dict(dynamic)
    dynamic["output_dir"] = str(tmp_dir)
    return {
        "global_parameters": {
            "N_points": N_POINTS, "w_max": W_MAX, "U": U, "flavors": ["up", "down"],
        },
        "solver": {
            "static": {
                "store": False, "ph_sym": False,
                "use_potthoff_band_shift": potthoff,
            },
            "modifiable": {"impurity_onsite_e": {"up": eps, "down": eps}},
            "dynamic": dynamic,
        },
    }

# --------------------------------------------------------------------------
# The semicircular hybridization itself (no solve needed)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "half_bandwidth, center", [(2.0, 0.0), (2.0, 0.7), (10.0, 0.0), (3.5, -1.3)]
)
def test_semicircular_hybridization_moments_are_analytic(half_bandwidth, center, tmp_path):
    """
    A semicircle of half-bandwidth D centred at c, normalized to unit weight,
    has exactly D_1 = 1, D_2 = c, D_3 = c^2 + D^2/4, and peak strength
    Gamma = -Im Delta^R(c) = 2/D.
    """
    dynamic = {
        "T": T,
        "Delta_D": half_bandwidth,
        "Delta_center": center,
        "mu": 0.0,
        "V": V,
    }
    solver = Solver(_wrap(dynamic, 0.0, False, tmp_path))

    for fl in solver.flavors:
        D = hybridization_moments(solver.w, solver.Delta[fl].R, kmax=3)
        assert D[1] == pytest.approx(1.0, abs=1e-4)
        assert D[2] == pytest.approx(center, abs=1e-3)
        assert D[3] == pytest.approx(center ** 2 + half_bandwidth ** 2 / 4, rel=1e-3)

        peak = -np.imag(solver.Delta[fl].R)[np.argmin(np.abs(solver.w - center))]
        assert peak == pytest.approx(2.0 / half_bandwidth, rel=1e-3)

        # Hard band edge: identically zero outside [c-D, c+D].
        outside = np.abs(solver.w - center) > half_bandwidth
        assert np.allclose(np.imag(solver.Delta[fl].R)[outside], 0.0)


def test_semicircular_accepts_per_flavor_parameters(tmp_path):
    dynamic = {
        "T": T,
        "Delta_D": {"up": 8.0, "down": 12.0},
        "Delta_center": {"up": 0.5, "down": -0.5},
        "mu": 0.0,
        "V": V,
    }
    solver = Solver(_wrap(dynamic, 0.0, False, tmp_path))

    for fl, center in (("up", 0.5), ("down", -0.5)):
        D = hybridization_moments(solver.w, solver.Delta[fl].R, kmax=2)
        assert D[1] == pytest.approx(1.0, abs=1e-4)
        assert D[2] == pytest.approx(center, abs=1e-3)


@pytest.mark.parametrize("eta", [0.0, 0.005, 0.01, 0.02])
def test_delta_eta_adds_a_flat_floor_of_known_weight(eta, tmp_path):
    """
    The broadening is a flat band spanning the grid, so it adds exactly
    2 * w_max * eta / pi to D_1 and, being symmetric, leaves D_2 alone.
    """
    dynamic = {"T": T, "Delta_D": 10.0, "Delta_center": 0.0, "Delta_eta": eta, "mu": 0.0, "V": V}
    solver = Solver(_wrap(dynamic, 0.0, False, tmp_path))

    for fl in solver.flavors:
        D = hybridization_moments(solver.w, solver.Delta[fl].R, kmax=2)
        assert D[1] == pytest.approx(1.0 + 2 * W_MAX * eta / np.pi, abs=1e-3)
        assert D[2] == pytest.approx(0.0, abs=1e-3)

        im_outside = np.imag(solver.Delta[fl].R)[np.abs(solver.w) > 10.0]
        assert np.allclose(im_outside, -eta)


def test_delta_eta_rescues_an_out_of_band_bound_state(tmp_path):
    """
    Half-bandwidth 2 with U=4 puts the upper Hubbard band near w = +5.9, outside
    the bath band, where a bare semicircle has Im Delta^R identically zero: the
    state becomes a delta function, int A(w) dw falls to ~0.97 on this grid and
    even the m=1 sum rule fails by ~14%. A small constant broadening gives it a
    finite width and everything is restored -- while leaving the plain scheme's
    m=3 defect fully intact, so the regularization does not mask the physics it
    is there to measure.
    """
    dynamic = {"T": T, "Delta_D": 2.0, "Delta_center": 0.0, "Delta_eta": 0.01, "mu": 0.0, "V": V}

    deviations = {}
    for potthoff in (False, True):
        solver = Solver(_wrap(dynamic, 0.0, potthoff, tmp_path))
        assert solver.solve().success

        report = check_sum_rules(solver)
        for fl in solver.flavors:
            moments = report[fl]["moments"]
            assert moments[0]["integrated"] == pytest.approx(1.0, abs=1e-3)
            assert moments[1]["rel_dev"] < 5e-4
            assert moments[2]["rel_dev"] < 5e-4
        deviations[potthoff] = report["up"]["moments"][3]["rel_dev"]

    assert deviations[False] > 5e-3
    assert deviations[True] < 1e-3
    assert deviations[False] / deviations[True] > 20.0


def test_delta_eta_rejects_negative_values(tmp_path):
    dynamic = {"T": T, "Delta_D": 10.0, "Delta_center": 0.0, "Delta_eta": -0.01, "mu": 0.0, "V": V}
    with pytest.raises(ValueError, match="Delta_eta"):
        Solver(_wrap(dynamic, 0.0, False, tmp_path))


def test_semicircular_requires_a_center(tmp_path):
    """'Delta_center' is mandatory alongside 'Delta_D', as it is alongside
    'Delta_gamma' for the Lorentzian -- pass 0 explicitly for a centred band."""
    dynamic = {"T": T, "Delta_D": 10.0, "mu": 0.0, "V": V}
    with pytest.raises(ValueError, match="Delta_center"):
        Solver(_wrap(dynamic, 0.0, False, tmp_path))


def test_semicircular_and_lorentzian_are_mutually_exclusive(tmp_path):
    dynamic = {
        "T": T,
        "Delta_D": 10.0,
        "Delta_center": 0.0,
        "Delta_gamma": 1.0,
        "mu": 0.0,
        "V": V,
    }
    with pytest.raises(ValueError, match="Ambiguous hybridization"):
        Solver(_wrap(dynamic, 0.0, False, tmp_path))


