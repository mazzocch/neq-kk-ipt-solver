"""
Example 3 -- far from half filling

The same bath again, now with the impurity level at eps = 0, far above the
particle-hole symmetric value. The occupation drops well below 1/2 and the
double occupancy is strongly suppressed. This is the regime in which the
Kajueter-Kotliar interpolation matters most: plain second-order perturbation
theory is poor away from half filling, which is what the ansatz is for.

Run:  python examples/03_away_from_half_filling.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POINTS = 10001
W_MAX = 50.0
U = 6.0
EPS = 0.0


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
        title=r"Away from half filling: $U=6$, $\epsilon=0$, box-shaped-DOS bath",
        subtitle=r"$D=10$, $\Gamma=1$, $T=0.05$, $\mu=0$, $V=0$ -- equilibrium, spin symmetric",
        filename="03_away_from_half_filling.png")


if __name__ == "__main__":
    main()
