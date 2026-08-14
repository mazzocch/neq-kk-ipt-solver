"""
Example 4 -- voltage-biased steady state

Example 2 driven out of equilibrium. The two leads are held at different
chemical potentials, mu_l = mu + V/2 and mu_r = mu - V/2 with V = 1.5, so the
hybridization acquires a genuinely non-thermal Keldysh component and the
steady state is not described by any equilibrium distribution. Compare the
double occupancy with Example 2: the bias suppresses it.

Run:  python examples/04_voltage_bias.py
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
                "mu": 0.0, "V": 1.5,
                "output_dir": output_dir,
            },
        },
    }


def main() -> None:
    from _common import run
    print(__doc__)
    run(build_input,
        title=r"Voltage bias: $U=6$, $\epsilon=-3.25$, $V=1.5$, flat-DOS bath",
        subtitle=r"$D=10$, $\Gamma=1$, $T=0.05$, $\mu=0$ -- nonequilibrium steady state",
        filename="04_voltage_bias.png")


if __name__ == "__main__":
    main()
