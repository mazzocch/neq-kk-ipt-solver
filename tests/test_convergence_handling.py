"""
Tests for the restart ladder and the non-convergence policy of Solver.solve().

The motivating failure is an outer self-consistency loop (e.g. DMFT): a solve
that stops without finding a root still leaves a Green's function and a
self-energy on the solver, and if those are consumed silently every subsequent
iteration is corrupted. The solver therefore (a) retries from a ladder of
alternative starting points, (b) judges convergence by the residual rather than
by the optimizer's own success flag, and (c) raises by default when nothing
converges.
"""
import numpy as np
import pytest

from neq_kk_ipt_solver import ConvergenceError, Solver

U = 6.0
T = 0.05
GAMMA = 1.0
CENTRES = {"up": 1.0, "down": -1.0}
FLAVORS = ["up", "down"]


def build(tmp_path, *, n_points: int = 2001, **static) -> dict:
    config = {
        "store": False,
        "spin_sym": False,
        "ph_sym": False,
        "use_potthoff_band_shift": False,
    }
    config.update(static)
    return {
        "global_parameters": {
            "N_points": n_points, "w_max": 50.0, "U": U, "flavors": list(FLAVORS),
        },
        "solver": {
            "static": config,
            "modifiable": {"impurity_onsite_e": {fl: -U / 2 for fl in FLAVORS}},
            "dynamic": {
                "T": T,
                "Delta_center": dict(CENTRES),
                "Delta_gamma": {fl: GAMMA for fl in FLAVORS},
                "mu": 0.0, "V": 0.0, "output_dir": str(tmp_path),
            },
        },
    }


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_defaults_are_five_restarts_and_raise(tmp_path):
    solver = Solver(build(tmp_path))
    assert solver.max_restarts == 5
    assert solver.on_convergence_failure == "raise"
    assert solver.residual_tol == pytest.approx(1e-6)


def test_invalid_failure_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="on_convergence_failure"):
        Solver(build(tmp_path, on_convergence_failure="explode"))


# --------------------------------------------------------------------------
# The happy path is untouched
# --------------------------------------------------------------------------

def test_successful_solve_uses_no_restarts(tmp_path):
    solver = Solver(build(tmp_path))
    sol = solver.solve()

    assert sol.success
    assert solver.solve_attempts == 1
    assert solver.solve_strategy == "initial guess"
    assert solver.solve_residual < solver.residual_tol


# --------------------------------------------------------------------------
# Failure policy
# --------------------------------------------------------------------------

def test_non_convergence_raises_by_default(tmp_path):
    """maxfev=2 makes it impossible for any attempt to find a root."""
    solver = Solver(build(tmp_path))
    with pytest.raises(ConvergenceError, match="did not converge"):
        solver.solve(maxfev=2)
    assert solver.solve_attempts == 6  # 1 initial + 5 restarts


def test_non_convergence_can_be_downgraded_to_a_warning(tmp_path):
    solver = Solver(build(tmp_path, on_convergence_failure="warn"))
    sol = solver.solve(maxfev=2)

    assert not sol.success
    assert solver.solve_residual > solver.residual_tol


def test_max_restarts_zero_makes_a_single_attempt(tmp_path):
    solver = Solver(build(tmp_path, max_restarts=0, on_convergence_failure="warn"))
    solver.solve(maxfev=2)
    assert solver.solve_attempts == 1


def test_failure_message_names_the_polluted_state(tmp_path):
    """The message has to say the stored G/Sigma are unusable, since that is the
    whole point of failing loudly."""
    solver = Solver(build(tmp_path))
    with pytest.raises(ConvergenceError) as excinfo:
        solver.solve(maxfev=2)
    assert "must not be fed into an outer self-consistency loop" in str(excinfo.value)


# --------------------------------------------------------------------------
# Convergence is judged by the residual, not by the optimizer's flag
# --------------------------------------------------------------------------

def test_success_requires_a_small_residual_not_just_the_optimizer_flag(tmp_path):
    """
    Guards a real bug: scipy's Levenberg-Marquardt reports success as soon as it
    stops making progress, which it will do at points that are not roots at all
    (observed at U=10 with |residual| ~ 1e-2 against ~1e-12 for the true
    solution). Driving residual_tol below anything achievable must therefore be
    reported as a failure even though the underlying algorithm succeeds.
    """
    solver = Solver(build(tmp_path, residual_tol=1e-300, on_convergence_failure="warn"))
    sol = solver.solve()

    assert sol.scipy_success is True     # the algorithm itself was happy
    assert sol.success is False          # but the residual gate was not
    assert solver.solve_residual > 0.0


def test_reported_success_always_matches_the_residual(tmp_path):
    solver = Solver(build(tmp_path, on_convergence_failure="warn"))
    sol = solver.solve()
    assert sol.success == (solver.solve_residual <= solver.residual_tol)


# --------------------------------------------------------------------------
# The restart ladder actually recovers something
# --------------------------------------------------------------------------

def test_restarts_recover_a_bad_initial_guess(tmp_path):
    """
    A strongly polarized starting guess sends the root-finder into the
    near-saturation region where the n(1-n) denominators blow up, and the solve
    fails outright with restarts disabled. With the ladder enabled it recovers,
    and must land on the same solution a cold start finds.
    """
    reference = Solver(build(tmp_path))
    assert reference.solve().success
    n_up_reference = float(np.real(reference.n_occ["up"]))

    # A near-saturated occupation combined with a displaced auxiliary potential.
    bad_guess = {"up": np.array([3.0, 0.97]), "down": np.array([-3.0, 0.03])}

    without = Solver(build(tmp_path, max_restarts=0, on_convergence_failure="warn"))
    without.x_guess = dict(bad_guess)
    failed = without.solve()

    with_restarts = Solver(build(tmp_path, on_convergence_failure="warn"))
    with_restarts.x_guess = dict(bad_guess)
    recovered = with_restarts.solve()

    if failed.success:
        pytest.skip("this starting guess no longer defeats the bare solver")

    assert recovered.success
    assert with_restarts.solve_attempts > 1
    assert float(np.real(with_restarts.n_occ["up"])) == pytest.approx(
        n_up_reference, abs=1e-4
    )


def test_restart_ladder_is_deterministic(tmp_path):
    """The jittered restarts are seeded, so two identical runs agree exactly."""
    guess = {"up": np.array([3.0, 0.97]), "down": np.array([-3.0, 0.03])}
    results = []
    for _ in range(2):
        solver = Solver(build(tmp_path, on_convergence_failure="warn"))
        solver.x_guess = dict(guess)
        solver.solve()
        results.append((solver.solve_attempts, solver.solve_strategy,
                        float(np.real(solver.n_occ["up"]))))
    assert results[0][:2] == results[1][:2]
    assert results[0][2] == pytest.approx(results[1][2], abs=1e-12)


# --------------------------------------------------------------------------
# Potthoff pre-solve
# --------------------------------------------------------------------------

def test_potthoff_flag_is_restored_if_the_presolve_fails(tmp_path):
    """
    A cold-started Potthoff run first solves the plain scheme to warm-start
    itself. If that pre-solve raises, the flag must not be left switched off.
    """
    solver = Solver(build(tmp_path, use_potthoff_band_shift=True))
    with pytest.raises(ConvergenceError):
        solver.solve(maxfev=2)
    assert solver.use_potthoff_band_shift is True
