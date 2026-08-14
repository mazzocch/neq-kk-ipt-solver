"""
Example 5 -- spin-dependent hybridization

A bath that is different for the two spin species: a Lorentzian of width
gamma = 1 centred at +1 for spin up and at -1 for spin down, each normalized
to unit spectral weight. The onsite energy is still spin independent and sits
at the particle-hole symmetric value, and the system is in equilibrium.

Even so the solution is strongly spin polarized. The two spins see identical
broadening at the Fermi level -- Gamma_up(0) = Gamma_down(0) exactly -- so the
polarization is driven entirely by the real part of the hybridization, which
acts as a static exchange field Re Delta^R_sigma(0) = -/+ 0.5.

This is the path to use for any per-flavor bath: every shape parameter accepts
a per-flavor dict in place of a single number.

Run:  python examples/05_spin_dependent_hybridization.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POINTS = 10001
W_MAX = 50.0
U = 3.0
EPS = -U / 2


def build_input(output_dir: str) -> dict:
    """The solver input for this example. Importable without matplotlib."""
    return {
        "global_parameters": {
            "N_points": N_POINTS, "w_max": W_MAX, "U": U,
            "flavors": ["up", "down"],
        },
        "solver": {
            "static": {
                "store": False,
                "spin_sym": False,
                "ph_sym": False,
            },
            "modifiable": {"impurity_onsite_e": {"up": EPS, "down": EPS}},
            "dynamic": {
                "T": 0.05,
                "Delta_center": {"up": 1.0, "down": -1.0},
                "Delta_gamma": {"up": 1.0, "down": 1.0},
                "mu": 0.0, "V": 0.0,
                "output_dir": output_dir,
            },
        },
    }


def main() -> None:
    from _common import run
    print(__doc__)
    run(build_input,
        title=r"Spin-dependent bath: $U=3$, $\epsilon=-U/2=-1.5$, Lorentzians at $\pm1$",
        subtitle=r"$\gamma=1$, $T=0.05$, $\mu=0$, $V=0$ -- equilibrium, spin resolved",
        filename="05_spin_dependent_hybridization.png")


if __name__ == "__main__":
    main()
