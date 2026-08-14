"""
Smoke tests for the scripts in examples/.

Examples rot: a rename or a signature change in the solver breaks them
silently, and nobody notices until a new user copies one. These tests import
each example, build its input, and solve it, so a breaking change shows up in
CI rather than in someone's first five minutes with the package.

They deliberately do NOT check physics -- the regression files do that -- and
they do not import matplotlib, which is an optional dependency. Each example
keeps its `build_input()` free of plotting imports precisely so this works.
The grid is coarsened to keep the suite fast; the examples themselves run on
the production grid.
"""
import importlib.util
import os
import sys

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
EXAMPLE_FILES = sorted(
    f for f in os.listdir(EXAMPLES_DIR)
    if f.endswith(".py") and not f.startswith("_")
)

# Coarse enough to be quick, wide enough that the solver still behaves.
SMOKE_N_POINTS = 1001


def load(filename: str):
    path = os.path.join(EXAMPLES_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_there_are_examples_to_check():
    """Guards against the discovery glob silently matching nothing."""
    assert len(EXAMPLE_FILES) >= 5


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_builds_and_solves(filename, tmp_path):
    from neq_kk_ipt_solver import Solver

    module = load(filename)
    assert module.__doc__, f"{filename} should document what it demonstrates"
    assert hasattr(module, "build_input"), f"{filename} must expose build_input()"
    assert hasattr(module, "main"), f"{filename} must expose main()"

    config = module.build_input(str(tmp_path))
    config["global_parameters"]["N_points"] = SMOKE_N_POINTS

    solver = Solver(config)
    sol = solver.solve()

    assert sol.success, f"{filename} did not converge"
    assert solver.solve_residual < 1e-8

    total = sum(float(solver.n_occ[fl].real) for fl in solver.flavors)
    assert 0.0 < total < 2.0
    assert solver.n_double >= -1e-9


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_is_runnable_from_any_directory(filename):
    """
    The examples insert their own directory on sys.path so that `from _common
    import run` works regardless of the working directory. Check that the path
    juggling actually holds, since it is easy to break by moving a file.
    """
    module = load(filename)
    assert EXAMPLES_DIR in sys.path
    assert os.path.isfile(os.path.join(EXAMPLES_DIR, "_common.py"))
