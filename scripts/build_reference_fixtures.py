"""
Rebuilds tests/data/*.json reference fixtures from the original IPT
solver_output files in the electric_field_hypercubic sister project,
including the full Green's function and self-energy arrays (not just scalar
occupations) so the physics regression tests can benchmark full spectral
data, not only a handful of observables.

Also recomputes n_double from those raw GF/SE arrays via the EXACT Keldysh
Galitskii-Migdal expression (Eq. 3 of the attached PDF excerpt, same formula
already implemented in neq_kk_ipt_solver.solver.Solver.calc_occupations()):

    n_d = -i/(2*pi*U) * integral dw [Sigma^R(w) G^<(w) + Sigma^<(w) G^A(w)]
    X^< = X^K/2 - i*Im(X^R),  G^A = (G^R)^*

rather than trusting the "N_double" field already stored in those files,
since a very early version of this solver used an approximate formula
(Eq. 2 of the same excerpt) that coincides with Eq. 3 only in equilibrium
(via the fluctuation-dissipation theorem) but not away from it. Comparing
the two below confirms exactly that: equilibrium matches to ~1e-14/1e-16,
nonequilibrium (V=1.0) differs by ~6-7e-5.

Not part of the installed package; a one-off fixture-generation script, run
once and not re-run automatically (the corrected reference values are then
simply committed as data).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "tests", "data")

EQ_SOURCES = {
    "eps0": "/Users/tommasomariamazzocchi/Desktop/projects/electric_field_hypercubic/preliminary_tests/march12/aim_T0p05_U5p5_eps0/solver_output_2026-03-12T15:58:31.json",
    "eps_m3": "/Users/tommasomariamazzocchi/Desktop/projects/electric_field_hypercubic/preliminary_tests/march12/aim_T0p05_U5p5_eps_m3/solver_output_2026-03-12T16:02:09.json",
}

NEQ_SOURCES = {
    "eps0": "/Users/tommasomariamazzocchi/Desktop/projects/electric_field_hypercubic/runs_kkipt_new/U4_T0p1175_eps0_ipt/solver_out_V1.000000/solver_output_2026-03-19T21:31:08.json",
    "eps_m2p25": "/Users/tommasomariamazzocchi/Desktop/projects/electric_field_hypercubic/runs_kkipt_new/U4_T0p1175_eps_m2p25_ipt/solver_out_V1.000000/solver_output_2026-03-20T08:50:43.json",
}


def recompute_n_double(results: dict, U: float, fl0: str = "up") -> float:
    w = np.array(results[fl0]["w"])
    GF_R = np.array(results[fl0]["GF_R_re"]) + 1j * np.array(results[fl0]["GF_R_im"])
    GF_K = np.array(results[fl0]["GF_K_re"]) + 1j * np.array(results[fl0]["GF_K_im"])
    SE_R = np.array(results[fl0]["SE_R_re"]) + 1j * np.array(results[fl0]["SE_R_im"])
    SE_K = np.array(results[fl0]["SE_K_re"]) + 1j * np.array(results[fl0]["SE_K_im"])

    GF_A = GF_R.conj()
    GF_lt = GF_K / 2 - 1j * np.imag(GF_R)
    SE_lt = SE_K / 2 - 1j * np.imag(SE_R)

    integrand = SE_R * GF_lt + SE_lt * GF_A
    return float(np.real(-1j / (2 * np.pi * U) * np.trapezoid(integrand, w)))


def gf_se_block(results: dict, fl: str) -> dict:
    r = results[fl]
    return {
        "GF_R_re": r["GF_R_re"], "GF_R_im": r["GF_R_im"],
        "GF_K_re": r["GF_K_re"], "GF_K_im": r["GF_K_im"],
        "SE_R_re": r["SE_R_re"], "SE_R_im": r["SE_R_im"],
        "SE_K_re": r["SE_K_re"], "SE_K_im": r["SE_K_im"],
    }


def build_equilibrium_fixture() -> dict:
    fixture = {}
    shared_delta = None
    for name, path in EQ_SOURCES.items():
        with open(path) as f:
            d = json.load(f)
        inp = d["input"]
        gp = inp["global_parameters"]
        onsite = inp["solver"]["modifiable"]["impurity_onsite_e"]
        dyn = inp["solver"]["dynamic"]
        r = d["results"]
        U = gp["U"]

        if shared_delta is None:
            shared_delta = {"Delta_R_im": dyn["Delta_R_im"], "Delta_K_im": dyn["Delta_K_im"]}

        n_double_recalc = recompute_n_double(r, U)
        stored = r["N_double"]
        print(f"[equilibrium/{name}] stored N_double={stored:.6f}, "
              f"recalculated (Eq.3)={n_double_recalc:.6f}, diff={abs(stored - n_double_recalc):.3e}")

        fixture[name] = {
            "global_parameters": gp,
            "impurity_onsite_e": onsite,
            "reference_gf_se": {fl: gf_se_block(r, fl) for fl in ("up", "down")},
            "expected": {
                "n_occ_up": r["up"]["n_occ"],
                "n_occ_down": r["down"]["n_occ"],
                "n_double": n_double_recalc,
                "n_double_stored_original": stored,
            },
        }

    fixture["shared_hybridization"] = shared_delta
    fixture["source"] = (
        "electric_field_hypercubic/preliminary_tests/march12 (single-band IPT, T=0.05, "
        "U=5.5, Bethe-lattice-like hybridization, two impurity levels eps=0 and eps=-3). "
        "n_double recomputed from the raw GF/SE arrays via the exact Keldysh "
        "Galitskii-Migdal expression, not taken from the stored 'N_double' field."
    )
    return fixture


def build_nonequilibrium_fixture() -> dict:
    fixture = {}
    for name, path in NEQ_SOURCES.items():
        with open(path) as f:
            d = json.load(f)
        inp = d["input"]
        gp = inp["global_parameters"]
        onsite = inp["solver"]["modifiable"]["impurity_onsite_e"]
        dyn = inp["solver"]["dynamic"]
        r = d["results"]
        U = gp["U"]

        n_double_recalc = recompute_n_double(r, U)
        stored = r["N_double"]
        print(f"[nonequilibrium/{name}] stored N_double={stored:.6f}, "
              f"recalculated (Eq.3)={n_double_recalc:.6f}, diff={abs(stored - n_double_recalc):.3e}")

        fixture[name] = {
            "global_parameters": gp,
            "impurity_onsite_e": onsite,
            "hybridization": {"Delta_R_im": dyn["Delta_R_im"], "Delta_K_im": dyn["Delta_K_im"]},
            "reference_gf_se": {fl: gf_se_block(r, fl) for fl in ("up", "down")},
            "expected": {
                "n_occ_up": r["up"]["n_occ"],
                "n_occ_down": r["down"]["n_occ"],
                "n_double": n_double_recalc,
                "n_double_stored_original": stored,
            },
        }

    fixture["source"] = (
        "electric_field_hypercubic/runs_kkipt_new (nonequilibrium single-band IPT, U=4, "
        "T=0.1175, voltage bias V=1.0, flat-DOS lead hybridization, two impurity levels "
        "eps=0 and eps=-2.25). n_double recomputed from the raw GF/SE arrays via the exact "
        "Keldysh Galitskii-Migdal expression, not taken from the stored 'N_double' field -- "
        "a very early version of this solver used an approximate formula (equal to the exact "
        "one only in equilibrium, via the fluctuation-dissipation theorem) that differs from "
        "the exact one by ~6-7e-5 for these nonequilibrium (V=1.0) cases."
    )
    return fixture


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    eq_fixture = build_equilibrium_fixture()
    with open(os.path.join(DATA_DIR, "aim_T0p05_U5p5_reference.json"), "w") as f:
        json.dump(eq_fixture, f, indent=2)

    neq_fixture = build_nonequilibrium_fixture()
    with open(os.path.join(DATA_DIR, "aim_U4_T0p1175_V1_nonequilibrium_reference.json"), "w") as f:
        json.dump(neq_fixture, f, indent=2)

    print("\nWrote both fixture files.")


if __name__ == "__main__":
    main()
