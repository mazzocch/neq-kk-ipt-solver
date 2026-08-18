"""
Example 2 -- slightly above half filling

The same bath as Example 1, with the impurity level pushed 0.25 below the
particle-hole symmetric value. This is the "near half filling" setup of the
preprint: the occupation rises a little above 1/2 and the spectral function
loses its symmetry about omega = 0, while the three-peak structure survives.

Run:  python examples/02_near_half_filling.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POINTS = 10001
W_MAX = 50.0
U = 6.0
EPS = -3.25


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
        title=r"Near half filling: $U=6$, $\epsilon=-3.25$, box-shaped-DOS bath",
        subtitle=r"$D=10$, $\Gamma=1$, $T=0.05$, $\mu=0$, $V=0$ -- equilibrium, spin symmetric",
        filename="02_near_half_filling.png")


if __name__ == "__main__":
    main()
