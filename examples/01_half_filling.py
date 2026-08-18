"""
Example 1 -- half filling

The particle-hole symmetric point. A box-shaped-DOS reservoir of
half-bandwidth D = 10 with t_l = t_r = 1/sqrt(2), so that Gamma = 1, at
T = 0.05 in equilibrium (mu = V = 0). With eps = -U/2 the impurity sits at
half filling, n_up = n_down = 1/2, and the spectral function is symmetric
about omega = 0 with a quasiparticle peak between the two Hubbard bands.

Run:  python examples/01_half_filling.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POINTS = 10001
W_MAX = 50.0
U = 6.0
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
                "spin_sym": True,
                "ph_sym": False,
            },
            "modifiable": {"impurity_onsite_e": {"up": EPS, "down": EPS}},
            "dynamic": {
                "T": 0.05,
                "T_fict": 0.5,
                "D_l": 10.0, "D_r": 10.0,
                "t_l": 1.0 / 2 ** 0.5, "t_r": 1.0 / 2 ** 0.5,
                "mu": 0.0, "V": 0.0,
                "output_dir": output_dir,
            },
        },
    }


def main() -> None:
    from _common import run
    print(__doc__)
    run(build_input,
        title=r"Half filling: $U=6$, $\epsilon=-U/2=-3$, box-shaped-DOS bath",
        subtitle=r"$D=10$, $\Gamma=1$, $T=0.05$, $\mu=0$, $V=0$ -- equilibrium, spin symmetric",
        filename="01_half_filling.png")


if __name__ == "__main__":
    main()
