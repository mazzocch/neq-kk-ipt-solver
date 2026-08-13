"""
Builds tests/data/spin_dependent_lorentzian_reference.json.

The fixture pins the equilibrium occupations and double occupancy for a
spin-dependent Lorentzian hybridization -- centre +1 for spin up, -1 for spin
down, gamma = 1 -- at two temperatures, for both the plain KK-IPT-n0 scheme and
the Potthoff m=3 band-shift correction.

The two existing physics regression fixtures both use spin-symmetric
hybridizations, so nothing in the suite would otherwise catch a regression in
the spin-resolved machinery: the sigma/sigma-bar index structure of the
self-energy, the per-flavor Delta, or the four-dimensional root solve. The U
values straddle the point where the majority spin changes over, which is where
the solution is most sensitive.

Regenerate from the repository root, and only deliberately:

    python scripts/build_spin_dependent_fixture.py
"""
import json
import os

import numpy as np

from neq_kk_ipt_solver import Solver

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "data",
                   "spin_dependent_lorentzian_reference.json")

U_VALUES = [4.0, 8.0]
TEMPERATURES = [0.02, 0.05]
GAMMA = 1.0
CENTRES = {"up": 1.0, "down": -1.0}
N_POINTS = 10001
W_MAX = 50.0
FLAVORS = ["up", "down"]


def build_input(U: float, T: float, potthoff: bool, output_dir: str) -> dict:
    return {
        "global_parameters": {
            "N_points": N_POINTS, "w_max": W_MAX, "U": U, "flavors": list(FLAVORS),
        },
        "solver": {
            "static": {
                "store": False, "spin_sym": False, "ph_sym": False,
                "use_potthoff_band_shift": potthoff,
            },
            "modifiable": {"impurity_onsite_e": {fl: -U / 2 for fl in FLAVORS}},
            "dynamic": {
                "T": T,
                "Delta_center": dict(CENTRES),
                "Delta_gamma": {fl: GAMMA for fl in FLAVORS},
                "mu": 0.0, "V": 0.0, "output_dir": output_dir,
            },
        },
    }


def main() -> None:
    fixture = {
        "setup": {
            "hybridization": "Lorentzian, spin-dependent",
            "Delta_center": dict(CENTRES), "Delta_gamma": GAMMA,
            "mu": 0.0, "V": 0.0,
            "N_points": N_POINTS, "w_max": W_MAX,
            "onsite_energy": "eps_up = eps_down = -U/2 (spin independent)",
            "start": "cold start, mu0=0 and n=1/2; no warm start across U or T",
        },
        "cases": {},
    }

    for T in TEMPERATURES:
        for potthoff in (False, True):
            scheme = "potthoff" if potthoff else "plain"
            for U in U_VALUES:
                solver = Solver(build_input(U, T, potthoff, "/tmp/spin_dependent_fixture"))
                sol = solver.solve()
                if not sol.success:
                    raise SystemExit(
                        f"{scheme} T={T} U={U} did not converge; refusing to pin it"
                    )

                key = f"{scheme}_T{T:g}_U{U:g}".replace(".", "p")
                fixture["cases"][key] = {
                    "U": U,
                    "T": T,
                    "scheme": scheme,
                    "n_occ_up": float(np.real(solver.n_occ["up"])),
                    "n_occ_down": float(np.real(solver.n_occ["down"])),
                    "n_double": float(solver.n_double),
                    "residual_norm": float(solver.solve_residual),
                }
                print(f"[{scheme:8s} T={T:g} U={U:4.1f}] "
                      f"n_up={solver.n_occ['up']:.6f} n_dn={solver.n_occ['down']:.6f} "
                      f"n_d={solver.n_double:.6f} |res|={solver.solve_residual:.2e}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"\nWrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
