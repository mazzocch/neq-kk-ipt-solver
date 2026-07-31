# neq-kk-ipt-solver

A nonequilibrium Kajueter-Kotliar Iterated Perturbation Theory (IPT) impurity
solver, extended to arbitrary filling and to genuine nonequilibrium (Keldysh)
steady states, with an optional Potthoff-Wegner-Nolting (Phys. Rev. B 55,
16132 (1997)) m=3 moment band-shift correction.

This is a standalone extraction of the single-band impurity solver used in
Mazzocchi, Werner, Aichhorn, Arrigoni, *A steady-state study of the
nonequilibrium properties of realistic materials: Application of the mixed
configuration approximation*, arXiv:2602.05664, and in the two-orbital
benchmarks of Mazzocchi, Werner, Aichhorn, Arrigoni, Phys. Rev. B **112**,
155127 (2025) [arXiv:2507.10717].

## Physics background

The solver implements the Kajueter-Kotliar interpolative self-energy ansatz
for the single-impurity Anderson model, generalized away from half-filling
and particle-hole symmetry, and formulated on the Keldysh contour so it
applies equally to equilibrium and genuine nonequilibrium (voltage- or
temperature-biased) steady states. Given a hybridization function Delta(w)
(retarded and Keldysh components) and Hubbard interaction U, it solves a
coupled 4-dimensional nonlinear root problem (auxiliary chemical potential
mu0 and occupation n, for each spin flavor) to determine the impurity
self-energy and Green's function self-consistently.

An optional correction (`use_potthoff_band_shift`) adds the nonequilibrium
Keldysh analogue of the Potthoff-Wegner-Nolting m=3 moment band-shift term
(Phys. Rev. B 55, 16132 (1997)) to the self-energy.

## Installation

```bash
git clone https://github.com/mazzocch/neq-kk-ipt-solver.git
cd neq-kk-ipt-solver
pip install .
```

For development (editable install + running the tests):

```bash
pip install -e ".[test]"
pytest
```

## Quickstart

```python
from neq_kk_ipt_solver import Solver

input_data = {
    "global_parameters": {
        "N_points": 4001,
        "w_max": 20.0,
        "U": 3.0,
        "flavors": ["up", "down"],
    },
    "solver": {
        "static": {
            "spin_sym": True,
            "ph_sym": True,
        },
        "modifiable": {},
        "dynamic": {
            # Flat-DOS (semi-elliptic-like) reservoir hybridization
            "T": 0.05,
            "T_fict": 0.05,
            "D_l": 10.0,
            "D_r": 10.0,
            "t_l": 1.0,
            "t_r": 1.0,
            "mu": 0.0,
            "V": 0.0,
            "output_dir": "./out",
        },
    },
}

solver = Solver(input_data)
sol = solver.solve()

print("Converged:", sol.success)
print("Occupations:", solver.n_occ)
print("Double occupancy:", solver.n_double)

solver.store_output()  # writes ./out/solver_output_<timestamp>.json
```

## Parameter reference

### `global_parameters`

| Key | Type | Required | Description |
|---|---|---|---|
| `N_points` | integer | yes | Number of points in the frequency grid. |
| `w_max` | number | yes | Maximum frequency; the grid spans `[-w_max, w_max]`. |
| `U` | number | yes | Hubbard interaction strength. |
| `flavors` | array of strings | yes | The two spin flavors, e.g. `["up", "down"]`. Exactly 2 are required. |
| `strict` | boolean | no | If `true`, raises on JSON-schema validation failure instead of warning. Default `false`. |
| `J`, `Up`, `Upp` | number | no | Accepted for schema compatibility with the parent multiorbital project (Hund's coupling / inter-orbital Kanamori terms), but **not used** by this single-band solver. |

### `solver.static`

| Key | Type | Default | Description |
|---|---|---|---|
| `store` | boolean | `true` | Whether `store_output()` actually writes anything. |
| `spin_sym` | boolean | `true` | Enforce spin symmetry (both flavors share the same onsite energy). |
| `ph_sym` | boolean | `true` | Enforce particle-hole symmetry (`impurity_onsite_e = -U/2` for both flavors, when combined with `spin_sym`). |
| `use_potthoff_band_shift` | boolean | `false` | Enable the Potthoff-Wegner-Nolting m=3 moment band-shift correction. `false` reproduces the plain KK-IPT-n0 scheme. |
| `band_shift_inner_maxiter` | integer | `30` | Max inner fixed-point iterations for the interacting band shift `B_tilde` (only used if the above is `true`). |
| `band_shift_mixing` | number | `0.5` | Linear mixing factor in (0, 1] for that inner iteration. |
| `band_shift_tol` | number | `1e-8` | Convergence tolerance (max change in `B_tilde`) for that inner iteration. |

### `solver.modifiable`

| Key | Type | Description |
|---|---|---|
| `impurity_onsite_e` | object | Per-flavor onsite energy, e.g. `{"up": -1.5, "down": -1.5}`. Set automatically when `ph_sym` and `spin_sym` are both `true`. |

### `solver.dynamic`

Exactly one of the following three hybridization specifications must be given, plus `output_dir`:

**1. Explicit hybridization arrays** (fully general):

| Key | Type | Description |
|---|---|---|
| `Delta_R_im` | object | Imaginary part of the retarded hybridization, per flavor (e.g. `{"up": [...], "down": [...]}`). The real part is reconstructed via Kramers-Kronig. |
| `Delta_K_im` | object | Imaginary part of the Keldysh hybridization, per flavor. |

**2. Lorentzian hybridization** (single number = spin-symmetric, or per-flavor dict = spin-dependent):

| Key | Type | Description |
|---|---|---|
| `Delta_center` | number or object | Center(s) of the Lorentzian. |
| `Delta_gamma` | number or object | Width(s) of the Lorentzian (must be nonzero). |
| `mu` | number | Chemical potential of the hybridization/bath. |
| `V` | number | Applied voltage bias between the two leads (`mu_l = mu + V/2`, `mu_r = mu - V/2`). |

**3. Flat-DOS reservoir** (fallback if neither of the above is given):

| Key | Type | Description |
|---|---|---|
| `T_fict` | number | Fictitious inverse temperature shaping the flat DOS. |
| `D_l`, `D_r` | number | Half-bandwidths of the left/right leads. |
| `t_l`, `t_r` | number | Hopping amplitudes to the left/right leads. |
| `mu`, `V` | number | As above. |

Common to all three:

| Key | Type | Description |
|---|---|---|
| `output_dir` | string | Directory `store_output()` writes results into. Required. |

## API overview

- `Solver(input_data)` — construct and validate a solver instance from the JSON-schema-checked input above.
- `Solver.solve(maxfev=1000, res_tol=1e-8)` — solves the coupled nonlinear system; returns the `scipy.optimize.OptimizeResult`. Populates `n_occ`, `n_double`, `GF`, `SE`, `G0`.
- `Solver.calc_occupations()` — (called internally by `solve()`) computes occupations and the exact Keldysh Galitskii-Migdal double occupancy, valid both in and out of equilibrium.
- `Solver.store_output()` — writes a JSON file with the full solution (Green's functions, self-energy, occupations, input parameters, git-commit provenance).
- `Solver.set_Delta_external(Delta)`, `Solver.set_onsite_energies_external(...)` — for embedding this solver inside a larger self-consistency loop (e.g. DMFT), where the hybridization/onsite energy are supplied externally each iteration instead of built once from `solver.dynamic`.

## Status

This is an early (v0.1.0) extraction from a larger research codebase. The
core physics has been validated against published benchmarks (see the papers
above); the packaging, test coverage, and documentation here are a minimum
viable version, not yet exhaustive.

## License

MIT — see [LICENSE](LICENSE).
