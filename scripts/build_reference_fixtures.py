"""
Rebuilds the tests/data/*.json + *.npz reference fixtures from the original
IPT solver_output files in the electric_field_hypercubic sister project,
including the full Green's function and self-energy arrays (not just scalar
occupations) so the physics regression tests can benchmark full spectral
data, not only a handful of observables.

Numeric arrays (hybridization, GF, SE) are stored in a companion compressed
.npz file rather than inline in the JSON: JSON text-encodes floats at ~20+
bytes each with no compression, whereas np.savez_compressed stores them as
raw (gzip-compressed) binary, which shrunk these two fixtures from ~25MB
combined down to a small fraction of that at full, unchanged resolution
(N_points=10001, both flavors, no downsampling). The JSON files keep only
small scalar/metadata (global_parameters, onsite energies, expected
occupations/double-occupancy, provenance).

Also recomputes n_double from those raw GF/SE arrays via the EXACT Keldysh
Galitskii-Migdal expression (the same formula already implemented in
neq_kk_ipt_solver.solver.Solver.calc_occupations()):

    n_d = -i/(2*pi*U) * integral dw [Sigma^R(w) G^<(w) + Sigma^<(w) G^A(w)]
    X^< = X^K/2 - i*Im(X^R),  G^A = (G^R)^*

rather than trusting the "N_double" field already stored in those files,
since a very early version of this solver used an approximate formula that
coincides with this exact one only in equilibrium (via the
fluctuation-dissipation theorem) but not away from it. Comparing the two
below confirms exactly that: equilibrium matches to ~1e-14/1e-16,
nonequilibrium (V=1.0) differs by ~6-7e-5.

Not part of the installed package; a one-off fixture-generation script, run
once and not re-run automatically (the corrected reference values are then
simply committed as data).

Requires the SOURCE_DATA_ROOT environment variable to point at a local
checkout of the (private, unpublished) source-data project -- not committed
here, since the actual path is local-machine-specific.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "tests", "data")

SOURCE_DATA_ROOT = os.environ.get("SOURCE_DATA_ROOT")
if not SOURCE_DATA_ROOT:
    raise SystemExit(
        "Set the SOURCE_DATA_ROOT environment variable to the local path of "
        "the private source-data project before running this script."
    )

EQ_SOURCES = {
    "eps0": os.path.join(SOURCE_DATA_ROOT, "preliminary_tests/march12/aim_T0p05_U5p5_eps0/solver_output_2026-03-12T15:58:31.json"),
    "eps_m3": os.path.join(SOURCE_DATA_ROOT, "preliminary_tests/march12/aim_T0p05_U5p5_eps_m3/solver_output_2026-03-12T16:02:09.json"),
}

NEQ_SOURCES = {
    "eps0": os.path.join(SOURCE_DATA_ROOT, "runs_kkipt_new/U4_T0p1175_eps0_ipt/solver_out_V1.000000/solver_output_2026-03-19T21:31:08.json"),
    "eps_m2p25": os.path.join(SOURCE_DATA_ROOT, "runs_kkipt_new/U4_T0p1175_eps_m2p25_ipt/solver_out_V1.000000/solver_output_2026-03-20T08:50:43.json"),
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


def gf_se_arrays(results: dict, fl: str, prefix: str, npz_arrays: dict) -> None:
    r = results[fl]
    for key in ("GF_R_re", "GF_R_im", "GF_K_re", "GF_K_im", "SE_R_re", "SE_R_im", "SE_K_re", "SE_K_im"):
        npz_arrays[f"{prefix}_{key}"] = np.array(r[key])


def build_equilibrium_fixture() -> tuple[dict, dict]:
    meta = {}
    npz_arrays = {}
    shared_delta_written = False

    for name, path in EQ_SOURCES.items():
        with open(path) as f:
            d = json.load(f)
        inp = d["input"]
        gp = inp["global_parameters"]
        onsite = inp["solver"]["modifiable"]["impurity_onsite_e"]
        dyn = inp["solver"]["dynamic"]
        r = d["results"]
        U = gp["U"]

        if not shared_delta_written:
            for fl in ("up", "down"):
                npz_arrays[f"hyb_Delta_R_im_{fl}"] = np.array(dyn["Delta_R_im"][fl])
                npz_arrays[f"hyb_Delta_K_im_{fl}"] = np.array(dyn["Delta_K_im"][fl])
            shared_delta_written = True

        n_double_recalc = recompute_n_double(r, U)
        stored = r["N_double"]
        print(f"[equilibrium/{name}] stored N_double={stored:.6f}, "
              f"recalculated (Eq.3)={n_double_recalc:.6f}, diff={abs(stored - n_double_recalc):.3e}")

        for fl in ("up", "down"):
            gf_se_arrays(r, fl, f"{name}_{fl}", npz_arrays)

        meta[name] = {
            "global_parameters": gp,
            "impurity_onsite_e": onsite,
            "expected": {
                "n_occ_up": r["up"]["n_occ"],
                "n_occ_down": r["down"]["n_occ"],
                "n_double": n_double_recalc,
            },
        }

    meta["source"] = (
        "Equilibrium single-band IPT, T=0.05, U=5.5, Bethe-lattice-like hybridization, "
        "two impurity levels eps=0 and eps=-3. Hybridization and GF/SE arrays are in the "
        "companion .npz file."
    )
    return meta, npz_arrays


def build_nonequilibrium_fixture() -> tuple[dict, dict]:
    meta = {}
    npz_arrays = {}

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

        for fl in ("up", "down"):
            npz_arrays[f"{name}_hyb_Delta_R_im_{fl}"] = np.array(dyn["Delta_R_im"][fl])
            npz_arrays[f"{name}_hyb_Delta_K_im_{fl}"] = np.array(dyn["Delta_K_im"][fl])
            gf_se_arrays(r, fl, f"{name}_{fl}", npz_arrays)

        meta[name] = {
            "global_parameters": gp,
            "impurity_onsite_e": onsite,
            "expected": {
                "n_occ_up": r["up"]["n_occ"],
                "n_occ_down": r["down"]["n_occ"],
                "n_double": n_double_recalc,
            },
        }

    meta["source"] = (
        "Nonequilibrium single-band IPT, U=4, T=0.1175, for the representative voltage bias "
        "V=1.0, flat-DOS lead hybridization, two impurity levels eps=0 and eps=-2.25. "
        "Hybridization and GF/SE arrays are in the companion .npz file."
    )
    return meta, npz_arrays


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    eq_meta, eq_arrays = build_equilibrium_fixture()
    with open(os.path.join(DATA_DIR, "aim_T0p05_U5p5_reference.json"), "w") as f:
        json.dump(eq_meta, f, indent=2)
    np.savez_compressed(os.path.join(DATA_DIR, "aim_T0p05_U5p5_reference.npz"), **eq_arrays)

    neq_meta, neq_arrays = build_nonequilibrium_fixture()
    with open(os.path.join(DATA_DIR, "aim_U4_T0p1175_V1_nonequilibrium_reference.json"), "w") as f:
        json.dump(neq_meta, f, indent=2)
    np.savez_compressed(os.path.join(DATA_DIR, "aim_U4_T0p1175_V1_nonequilibrium_reference.npz"), **neq_arrays)

    print("\nWrote both fixture (.json metadata + .npz arrays) file pairs.")


if __name__ == "__main__":
    main()
