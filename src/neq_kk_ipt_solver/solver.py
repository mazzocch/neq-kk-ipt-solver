"""
Nonequilibrium Kajueter-Kotliar Iterated Perturbation Theory (IPT) impurity
solver, extended to arbitrary filling and to genuine nonequilibrium (Keldysh)
steady states, with an optional Potthoff-Wegner-Nolting [Phys. Rev. B 55,
16132 (1997)] m=3 moment band-shift correction.

Physics background and validation: see preprint on the nonequilibrium
IPT extension referenced in the README [https://doi.org/10.48550/arXiv.2604.15942].

Author: Tommaso Maria Mazzocchi
"""

import json
import numpy as np
import datetime
import os
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from time import time
from copy import deepcopy
from scipy.signal import convolve
from scipy.optimize import root

from .utils import Keldysh, set_global_parameters, KK, fermi, get_git_commit_hash


class ConvergenceError(RuntimeError):
    """
    Raised when the coupled root solve fails on every restart attempt.

    A non-converged solve leaves G, Sigma and the occupations at whatever point
    the root-finder stopped at. Those are not a solution of the impurity
    problem, and feeding them into an outer loop (e.g. DMFT) silently poisons
    every subsequent iteration. Failing loudly is therefore the default; set
    'on_convergence_failure' to 'warn' to get the old behaviour of returning the
    non-converged result instead.
    """


class Solver:

    def __init__(self, input_data: dict) -> None:
        if "solver" not in input_data.keys():
            raise ValueError("ERROR: No input parameters for the solver in input JSON file!")

        set_global_parameters(self, input_data)

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_data = input_data

        now = datetime.datetime.now()
        formatted_now = now.replace(microsecond=0).isoformat()

        self.default_config_static = {
            "spin_sym": True,
            "ph_sym": True,
            "store": True,
            "use_potthoff_band_shift": False,
            "band_shift_inner_maxiter": 30,
            "band_shift_mixing": 0.5,
            "band_shift_tol": 1e-8,
            "max_restarts": 5,
            "on_convergence_failure": "raise",
            "residual_tol": 1e-6,
        }

        self.filename = f"solver_output_{formatted_now}.json"
        self.n_double = None
        self.n_occ = {}
        self.n0_occ = {}
        self.impurity_onsite_e = {}
        self.results = {}

        self._check_config()
        self._load_config()

        self.G0 = {
            fl: Keldysh(
                np.zeros(self.N_points, dtype=complex),
                np.zeros(self.N_points, dtype=complex)
            )
            for fl in self.flavors
        }
        self.GF = {
            fl: Keldysh(
                np.zeros(self.N_points, dtype=complex),
                np.zeros(self.N_points, dtype=complex)
            )
            for fl in self.flavors
        }
        self.SE = {
            fl: Keldysh(
                np.zeros(self.N_points, dtype=complex),
                np.zeros(self.N_points, dtype=complex)
            )
            for fl in self.flavors
        }
        self.IPT_diag = {
            fl: Keldysh(
                np.zeros(self.N_points, dtype=complex),
                np.zeros(self.N_points, dtype=complex)
            )
            for fl in self.flavors
        }
        self.Delta = {
            fl: Keldysh(
                np.zeros(self.N_points, dtype=complex),
                np.zeros(self.N_points, dtype=complex)
            )
            for fl in self.flavors
        }

        # Potthoff-Wegner-Nolting (PRB 55, 16132) m=3 moment band-shift correction.
        # B0_tilde is the Weiss/HF-level band shift, B_tilde its interacting counterpart;
        # both are only populated when use_potthoff_band_shift is True.
        self.B0_tilde = {fl: 0.0 for fl in self.flavors}
        self.B_tilde = {fl: 0.0 for fl in self.flavors}
        self.alpha = {fl: np.nan for fl in self.flavors}
        self.beta = {fl: np.nan for fl in self.flavors}
        self.band_shift_error = 0.0
        self.band_shift_iterations = 0
        self.band_shift_diverged_count = 0

        # Populated by solve(): how many starting points were tried and which
        # one produced the returned solution.
        self.solve_attempts = 0
        self.solve_strategy = None
        self.solve_residual = np.inf

        if self.on_convergence_failure not in ("raise", "warn"):
            raise ValueError(
                "ERROR: 'on_convergence_failure' must be 'raise' or 'warn', got "
                f"{self.on_convergence_failure!r}."
            )

        if "dynamic" in input_data["solver"].keys():
            self._set_Delta()

    def _check_config(self) -> None:
        with open(
            os.path.join(self.dir_path, "schemas", "solver.schema.json"), "r"
        ) as file:
            json_schema = json.load(file)

        try:
            validate(instance=self.input_data["solver"], schema=json_schema)
            print("\n\nImpurity solver input JSON data is VALID.\n\n")
        except ValidationError as e:
            if self.strict:
                raise ValueError("\n\nERROR: Impurity solver input JSON data is INVALID!!\n\n")
            else:
                print("\n\nWARNING: Impurity solver input JSON data is INVALID!!\n\n")
            print("Validation error:", e.message)

    def _load_config(self) -> None:
        """
        Validates the input data and sets parameters which are defined indirectly.
        """

        self.default_config_static.update(self.input_data["solver"]["static"])

        for key, value in self.default_config_static.items():
            setattr(self, key, value)

        if "impurity_onsite_e" in self.input_data["solver"]["modifiable"].keys():
            if (
                self.flavors[0] not in self.input_data["solver"]["modifiable"]["impurity_onsite_e"]
                or self.flavors[1] not in self.input_data["solver"]["modifiable"]["impurity_onsite_e"]
            ):
                raise ValueError("ERROR: Flavors' keys are incorrect or missing!")
            else:
                self.impurity_onsite_e = {
                    fl: self.input_data["solver"]["modifiable"]["impurity_onsite_e"][fl]
                    for fl in self.flavors
                }

                if self.impurity_onsite_e[self.flavors[0]] != self.impurity_onsite_e[self.flavors[1]]:
                    self.spin_sym = False

                if self.impurity_onsite_e[self.flavors[0]] != - self.U / 2:
                    self.ph_sym = False

        for key, value in self.input_data["solver"]["modifiable"].items():
            setattr(self, key, value)

        if "dynamic" in self.input_data["solver"].keys():
            for key, value in self.input_data["solver"]["dynamic"].items():
                setattr(self, key, value)

        self.w = np.linspace(-self.w_max, self.w_max, self.N_points)

        if self.ph_sym and self.spin_sym:
            for fl in self.flavors:
                self.impurity_onsite_e[fl] = -self.U / 2

    def _clip_occ(self, n: float) -> float:
        return float(np.clip(n, 1e-8, 1 - 1e-8))

    def _other_flavor(self, fl: str) -> str:
        if len(self.flavors) != 2:
            raise ValueError("This implementation assumes exactly 2 flavors.")
        return self.flavors[1] if fl == self.flavors[0] else self.flavors[0]

    def _lesser(self, G: Keldysh) -> np.ndarray:
        return G.K / 2.0 - 1j * np.imag(G.R)

    def _greater(self, G: Keldysh) -> np.ndarray:
        return G.K / 2.0 + 1j * np.imag(G.R)

    def _as_flavor_dict(self, value, name: str) -> dict[str, float]:
        """Accept either a single number, shared by both flavors, or a per-flavor dict."""
        if isinstance(value, dict):
            missing = [fl for fl in self.flavors if fl not in value]
            if missing:
                raise ValueError(f"ERROR: Missing '{name}' entries for flavors: {missing}")
            return {fl: float(value[fl]) for fl in self.flavors}
        return {fl: float(value) for fl in self.flavors}

    def _pack_x(self, mu0_dict: dict[str, float], n_dict: dict[str, float]) -> np.ndarray:
        vals = []
        for fl in self.flavors:
            vals.extend([float(mu0_dict[fl]), self._clip_occ(float(n_dict[fl]))])
        return np.array(vals, dtype=float)

    def _unpack_x(self, x: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        if len(x) != 2 * len(self.flavors):
            raise ValueError("Input vector has wrong size for the number of flavors.")

        mu0_dict = {}
        n_dict = {}

        for i, fl in enumerate(self.flavors):
            mu0_dict[fl] = float(x[2 * i])
            n_dict[fl] = self._clip_occ(float(x[2 * i + 1]))

        return mu0_dict, n_dict

    def store_output(self) -> None:
        if not self.store:
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for fl in self.flavors:
            self.results[fl] = {
                "w": self.w.tolist(),
                "GF_R_re": np.real(self.GF[fl].R).tolist(),
                "GF_R_im": np.imag(self.GF[fl].R).tolist(),
                "GF_K_re": np.real(self.GF[fl].K).tolist(),
                "GF_K_im": np.imag(self.GF[fl].K).tolist(),
                "SE_R_re": np.real(self.SE[fl].R).tolist(),
                "SE_R_im": np.imag(self.SE[fl].R).tolist(),
                "SE_K_re": np.real(self.SE[fl].K).tolist(),
                "SE_K_im": np.imag(self.SE[fl].K).tolist(),
                "n_occ": self.n_occ[fl],
                "alpha": self.alpha[fl],
                "beta": self.beta[fl],
                "B_tilde": self.B_tilde[fl],
                "B0_tilde": self.B0_tilde[fl],
            }

        self.results["N_double"] = self.n_double
        self.results["use_potthoff_band_shift"] = self.use_potthoff_band_shift
        self.results["band_shift_error"] = self.band_shift_error
        self.results["band_shift_iterations"] = self.band_shift_iterations

        # Convergence provenance. Without this an output file gives no way to
        # tell a solution from a non-converged point, which is exactly the
        # failure mode the restart/raise machinery exists to prevent: a caller
        # that catches ConvergenceError and stores anyway would otherwise write
        # an unmarked, unusable result to disk.
        self.results["converged"] = bool(self.solve_residual <= self.residual_tol)
        self.results["residual_norm"] = float(self.solve_residual)
        self.results["residual_tol"] = float(self.residual_tol)
        self.results["solve_attempts"] = int(self.solve_attempts)
        self.results["solve_strategy"] = self.solve_strategy

        input_params = deepcopy(self.input_data)
        input_params = {
            key: value
            for key, value in input_params.items()
            if key in ["global_parameters", "solver"]
        }

        onsite_energy = {
            "impurity_onsite_e": {
                f"{fl}": self.impurity_onsite_e[fl]
                for fl in self.flavors
            },
        }

        if "solver" in input_params:
            if "modifiable" in input_params["solver"]:
                input_params["solver"]["modifiable"].update(onsite_energy)

            if "dynamic" in input_params["solver"]:
                input_params["solver"]["dynamic"] = {
                    "Delta_R_im": {
                        f"{fl}": np.imag(Delta.R).tolist()
                        for fl, Delta in self.Delta.items()
                    },
                    "Delta_K_im": {
                        f"{fl}": np.imag(Delta.K).tolist()
                        for fl, Delta in self.Delta.items()
                    },
                    "output_dir": self.output_dir
                }

        data = {
            "input": input_params,
            "results": self.results,
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "git_commit": get_git_commit_hash(),
                "script_file": os.path.basename(__file__)
            },
        }

        with open(f"{self.output_dir}/{self.filename}", "w") as file:
            json.dump(data, file, indent=4)

    def set_spin_symmetry(self, spin_sym: bool) -> None:
        self.spin_sym = spin_sym

    def set_onsite_energies_external(self, impurity_onsite_e: dict[str, float]) -> None:
        self.impurity_onsite_e = impurity_onsite_e

    def set_Delta_external(self, Delta: Keldysh) -> None:
        self.Delta = Delta

    def set_output_external(self, directory: str, filename: str) -> None:
        self.output_dir = directory
        self.filename = filename

    def _set_Delta(self) -> None:
        """
        Builds the hybridization function Delta, in order of precedence:
          1. explicit 'Delta_R_im'/'Delta_K_im' arrays (fully general, per flavor);
          2. a semi-elliptic ("semicircular") band of half-bandwidth 'Delta_D'
             centered on 'Delta_center' (both required, as for the Lorentzian);
          3. a Lorentzian of center 'Delta_center' and width 'Delta_gamma';
          4. the flat-DOS fallback built from 'T_fict', 'D_l', 'D_r', 't_l', 't_r'.

        For 2. and 3., each shape parameter may be a single number
        (spin-symmetric/degenerate hybridization, shared by both flavors) or a
        per-flavor dict (spin-dependent hybridization).
        """
        try:
            flat_DOS = False
            _ = self.Delta_R_im
        except Exception:
            flat_DOS = True

        semicircular_hyb = False
        lorentzian_hyb = False
        if flat_DOS:
            semicircular_hyb = hasattr(self, "Delta_D")
            try:
                _ = self.Delta_center
                _ = self.Delta_gamma
                lorentzian_hyb = True
            except Exception:
                lorentzian_hyb = False

        if semicircular_hyb and lorentzian_hyb:
            raise ValueError(
                "ERROR: Ambiguous hybridization request -- 'Delta_D' (semicircular) and "
                "'Delta_gamma' (Lorentzian) were both given. Provide exactly one."
            )

        if semicircular_hyb:
            print(
                "Semicircular hybridization requested via 'Delta_D'. Pass a single number "
                "for a spin-symmetric (degenerate) hybridization, or a per-flavor dict for "
                "a spin-dependent one."
            )

            # Same normalization convention as the Lorentzian branch below: unit
            # total spectral weight, i.e. -(1/pi) int dw Im Delta^R = 1, with the
            # shape parameter setting the width. The peak hybridization strength
            # is then Gamma = -Im Delta^R(center) = 2 / Delta_D, so a semicircle
            # matching a given Gamma needs Delta_D = 2/Gamma. For an arbitrary
            # normalization, supply 'Delta_R_im'/'Delta_K_im' directly instead.
            if not hasattr(self, "Delta_center"):
                raise ValueError(
                    "ERROR: The semicircular hybridization requires 'Delta_center' alongside "
                    "'Delta_D' (pass 0 for a band centered on zero), just as the Lorentzian "
                    "requires 'Delta_center' alongside 'Delta_gamma'."
                )

            centers = self._as_flavor_dict(self.Delta_center, "Delta_center")
            half_bandwidths = self._as_flavor_dict(self.Delta_D, "Delta_D")
            etas = self._as_flavor_dict(getattr(self, "Delta_eta", 0.0), "Delta_eta")

            mu_l = self.mu + self.V / 2
            mu_r = self.mu - self.V / 2

            for fl in self.flavors:
                D = half_bandwidths[fl]
                if D <= 1e-14:
                    raise ValueError(f"ERROR: 'Delta_D' for flavor '{fl}' must be positive.")

                eta = etas[fl]
                if eta < 0.0:
                    raise ValueError(
                        f"ERROR: 'Delta_eta' for flavor '{fl}' must be non-negative "
                        "(it is subtracted from Im Delta^R, which causality requires to be <= 0)."
                    )

                x = self.w - centers[fl]
                inside = np.abs(x) <= D

                Delta_R_im = np.zeros_like(self.w)
                Delta_R_im[inside] = -(2.0 / D ** 2) * np.sqrt(D ** 2 - x[inside] ** 2)

                # Optional constant broadening. A bare semicircle has Im Delta^R
                # exactly zero outside the band, so any impurity feature landing
                # there (typically a Hubbard band, once U is comparable to the
                # bandwidth) becomes a true bound state -- a delta function the
                # frequency grid cannot represent, which silently loses spectral
                # weight and breaks the sum rules. Subtracting a small constant
                # gives those states a finite, resolvable width.
                #
                # It is not free: the floor is a weak flat band spanning the whole
                # grid, so it adds 2 * w_max * eta / pi to the hybridization weight
                # D_1 (0.19 for eta=0.01 at w_max=30, i.e. ~19% on top of the
                # semicircle's own unit weight) and makes D_1 depend on w_max.
                # D_2 is unaffected, the floor being symmetric. Both are integrated
                # from the actual Delta, so the sum rules stay exact either way --
                # what changes is the model, not its consistency. Default 0.
                Delta_R_im -= eta

                Delta_R, _ = KK(self.w, Delta_R_im)
                Delta_K = 2j * np.imag(Delta_R) * (1 - fermi(self.w, mu_l, self.T) - fermi(self.w, mu_r, self.T))

                self.Delta[fl].R = Delta_R
                self.Delta[fl].K = Delta_K

            return

        if lorentzian_hyb:
            print(
                "Lorentzian hybridization requested via 'Delta_center'/'Delta_gamma'. "
                "Pass a single number for a spin-symmetric (degenerate) hybridization, "
                "or a per-flavor dict for a spin-dependent one."
            )

            centers = self._as_flavor_dict(self.Delta_center, "Delta_center")
            gammas = self._as_flavor_dict(self.Delta_gamma, "Delta_gamma")

            mu_l = self.mu + self.V / 2
            mu_r = self.mu - self.V / 2

            for fl in self.flavors:
                gamma = gammas[fl]
                if abs(gamma) < 1e-14:
                    raise ValueError(f"ERROR: 'Delta_gamma' for flavor '{fl}' must be nonzero.")

                x = self.w - centers[fl]
                Delta_R_im = - gamma / (x ** 2 + gamma ** 2)
                Delta_R, _ = KK(self.w, Delta_R_im)
                Delta_K = 2j * np.imag(Delta_R) * (1 - fermi(self.w, mu_l, self.T) - fermi(self.w, mu_r, self.T))

                self.Delta[fl].R = Delta_R
                self.Delta[fl].K = Delta_K

            return

        if flat_DOS:
            print("No hybridization has been given. A flat DOS with fictitious inverse temperature and fixed bandwidth is being used.")

        for fl in self.flavors:
            if flat_DOS:
                mu_l = self.mu + self.V / 2
                mu_r = self.mu - self.V / 2

                Lead_R_r_im = - (1 - fermi(self.w, -self.D_r, self.T_fict)) * fermi(self.w, self.D_r, self.T_fict)
                Lead_R_l_im = - (1 - fermi(self.w, -self.D_l, self.T_fict)) * fermi(self.w, self.D_l, self.T_fict)

                Lead_K_r = 2j * Lead_R_r_im * (1 - 2 * fermi(self.w, mu_r, self.T))
                Lead_K_l = 2j * Lead_R_l_im * (1 - 2 * fermi(self.w, mu_l, self.T))

                Delta_R_im = self.t_l ** 2 * Lead_R_l_im + self.t_r ** 2 * Lead_R_r_im
                Delta_R, _ = KK(self.w, Delta_R_im)
                Delta_K = self.t_l ** 2 * Lead_K_l + self.t_r ** 2 * Lead_K_r

                self.Delta[fl].R = Delta_R
                self.Delta[fl].K = Delta_K
            else:
                if f"{fl}" not in self.Delta_R_im.keys():
                    raise ValueError("ERROR: You have to provide at least the diagonal contributions of the hybridization in the flavors.")

                self.Delta[fl] = Keldysh(
                    1j * np.array(self.Delta_R_im[f"{fl}"]),
                    1j * np.array(self.Delta_K_im[f"{fl}"]),
                )
                self.Delta[fl].R, _ = KK(self.w, np.imag(self.Delta[fl].R))

    def calc_G0(self, mu0: float, fl: str) -> None:
        """
        Constructs the Weiss field (G0) from a given hybridization function according
        to the modified IPT scheme in Phys. Rev. Lett. 77 131 (1996).
        It assumes the auxiliary potential 'mu0' has been passed correctly,
        i.e. it corresponds to the current flavor 'fl' handled in this function.

        Written as the literal Dyson equation G0 = (G0^-1)^-1, with
        G0^-1 = (w + mu0) - Delta (Delta plays the role of a self-energy here).
        """
        G0_inv = Keldysh(R=self.w + mu0, K=0) - self.Delta[fl]
        self.G0[fl] = G0_inv.inverse()

        self.G0[fl].calc_spectrum()
        self.G0[fl].calc_distribution()
        self.G0[fl].calc_occupation()

    def _calc_IPT_diagram(self, enforce_anti_herm: bool = True) -> None:
        """
        Computes the 2nd-order IPT diagram for all flavors from the current Weiss field (G0).
        """
        dw = abs(self.w[1] - self.w[0])
        factor = (dw / (2 * np.pi)) ** 2 * self.U ** 2

        for fl in self.flavors:
            if len(self.flavors) != 2:
                raise ValueError("ERROR: This implementation assumes 2 flavors (spin up/down).")

            other = self._other_flavor(fl)

            G0s_gtr = self.G0[fl].K / 2.0 + 1j * np.imag(self.G0[fl].R)
            G0s_lss = self.G0[fl].K / 2.0 - 1j * np.imag(self.G0[fl].R)

            G0o_gtr = self.G0[other].K / 2.0 + 1j * np.imag(self.G0[other].R)
            G0o_lss = self.G0[other].K / 2.0 - 1j * np.imag(self.G0[other].R)

            diag_gtr = convolve(
                convolve(G0s_gtr, G0o_gtr, mode="same"),
                G0o_lss[::-1],
                mode="same",
            ) * factor

            diag_lss = convolve(
                convolve(G0s_lss, G0o_lss, mode="same"),
                G0o_gtr[::-1],
                mode="same",
            ) * factor

            if enforce_anti_herm:
                diag_gtr = 1j * diag_gtr.imag
                diag_lss = 1j * diag_lss.imag

            diag_R_imag = 0.5 * (diag_gtr - diag_lss).imag
            diag_K = diag_gtr + diag_lss

            diag_R, _ = KK(self.w, diag_R_imag)

            self.IPT_diag[fl].R = diag_R
            self.IPT_diag[fl].K = diag_K

    def calc_SE(self, mu0: float, n0_bar: float, n_bar: float, fl: str, recompute_ipt: bool = True) -> None:
        """
        Computes electronic self-energy (SE) using the generalized IPT Ansatz.
        The subscript 'bar' denotes the quantities of the opposite spin species.
        When no subscript is present it is implied that the quantity
        corresponding to the current flavor 'fl' has been passed correctly.

        If 'use_potthoff_band_shift' is True, beta (the coefficient 'B' below) is
        corrected by the nonequilibrium analogue of the Potthoff-Wegner-Nolting
        (PRB 55, 16132) m=3 moment term, B_tilde[other] - B0_tilde[other], which
        must have been (re)computed beforehand, e.g. in '_build_state'.
        """
        if recompute_ipt:
            self._calc_IPT_diagram()

        num_A = n_bar * (1 - n_bar) + 1e-12
        den_A = n0_bar * (1 - n0_bar) + 1e-12

        A = num_A / den_A
        beta_num = (1 - n_bar) * self.U + self.impurity_onsite_e[fl] + mu0

        if self.use_potthoff_band_shift:
            other = self._other_flavor(fl)
            beta_num += self.B_tilde[other] - self.B0_tilde[other]

        B = beta_num / (den_A * self.U ** 2)

        self.alpha[fl] = A
        self.beta[fl] = B

        SE_K_den = 1 - B * self.IPT_diag[fl].R + 1e-12
        SE_K_num = (
            A * self.IPT_diag[fl].K
            + B * self.IPT_diag[fl].K
            * (1 / (1 - B * np.conj(self.IPT_diag[fl].R)))
            * A * np.conj(self.IPT_diag[fl].R)
        )

        self.SE[fl].R = n_bar * self.U + A * self.IPT_diag[fl].R / (1 - B * self.IPT_diag[fl].R + 1e-12)
        self.SE[fl].K = SE_K_num / SE_K_den

        self.SE[fl].calc_spectrum()
        self.SE[fl].calc_distribution()
        self.SE[fl].calc_occupation()

    def calc_GF(self, mu0: float, fl: str) -> None:
        """
        Given the auxiliary chemical potential 'mu0' and flavor 'fl'
        computes the interacting impurity Green's function (GF) from Weiss field (G0)
        and electronic self-energy (SE) for the corresponding flavor.

        Written as the literal Dyson equation G = (G0^-1 - shift - Sigma)^-1,
        where 'shift' (mu0 + impurity_onsite_e) is the auxiliary-potential
        correction that must be subtracted back out of G0's own inverse
        (see calc_G0's docstring).
        """
        shift = mu0 + self.impurity_onsite_e[fl]
        GF_inv = self.G0[fl].inverse() - shift - self.SE[fl]
        self.GF[fl] = GF_inv.inverse()

        self.GF[fl].calc_spectrum()
        self.GF[fl].calc_distribution()
        self.GF[fl].calc_occupation()

    def _calc_band_shift_weiss(self, fl: str, *, n0_fl: float, n0_bar: float) -> float:
        """
        Weiss/Hartree-Fock-level band shift B0_tilde_fl, the nonequilibrium Keldysh
        analogue of Potthoff-Wegner-Nolting (PRB 55, 16132) Eq. (13)/(29):

            B0_tilde_fl = eps_fl
                + (2 n0_bar - 1) / [n0_fl(1-n0_fl)]
                  * (-i) int dw/(2 pi) [G0_fl^R Delta_fl^< + G0_fl^< Delta_fl^A]

        Its value is combined with the interacting counterpart 'B_tilde' (see
        '_calc_band_shift_interacting') and enters the self-energy of the
        opposite flavor via 'calc_SE'.
        """
        n0_fl = self._clip_occ(n0_fl)
        n0_bar = self._clip_occ(n0_bar)

        G0_lss = self._lesser(self.G0[fl])
        Delta_lss = self._lesser(self.Delta[fl])
        Delta_A = np.conj(self.Delta[fl].R)

        mixed_lss = self.G0[fl].R * Delta_lss + G0_lss * Delta_A
        mixed_corr = -1j * np.trapezoid(mixed_lss, self.w) / (2.0 * np.pi)

        den = n0_fl * (1.0 - n0_fl) + 1e-12
        B0_tilde = self.impurity_onsite_e[fl] + (2.0 * n0_bar - 1.0) * mixed_corr / den

        return float(np.real(B0_tilde))

    def _calc_band_shift_interacting(self, fl: str, *, n_fl: float) -> float:
        """
        Interacting band shift B_tilde_fl, the nonequilibrium Keldysh analogue of
        Potthoff-Wegner-Nolting (PRB 55, 16132) Eq. (44) with Q_fl = 2 Sigma_fl/U - 1:

            n_fl(1-n_fl) B_tilde_fl = n_fl(1-n_fl) eps_fl
                - i int dw/(2 pi) [Q_fl G_fl Delta_fl]^<

        Requires SE[fl] and GF[fl] to already correspond to the current trial state.
        """
        if abs(self.U) < 1e-12:
            raise ValueError("Potthoff band-shift correction requires finite U.")

        n_fl = self._clip_occ(n_fl)

        G_lss = self._lesser(self.GF[fl])
        Delta_lss = self._lesser(self.Delta[fl])
        SE_lss = self._lesser(self.SE[fl])

        Q_R = 2.0 * self.SE[fl].R / self.U - 1.0
        Q_lss = 2.0 * SE_lss / self.U

        G_A = np.conj(self.GF[fl].R)
        Delta_A = np.conj(self.Delta[fl].R)

        X_lss = (
            Q_R * self.GF[fl].R * Delta_lss
            + Q_R * G_lss * Delta_A
            + Q_lss * G_A * Delta_A
        )

        correction = -1j * np.trapezoid(X_lss, self.w) / (2.0 * np.pi)

        den = n_fl * (1.0 - n_fl) + 1e-12
        B_tilde = self.impurity_onsite_e[fl] + correction / den

        return float(np.real(B_tilde))

    def _build_state(self, mu0_all: dict[str, float], n_all: dict[str, float]) -> dict[str, float]:
        """
        Given trial values of mu0 and n for all flavors, builds G0, IPT diagram,
        SE, GF consistently for all flavors. Returns n0_all.

        If 'use_potthoff_band_shift' is True, an inner fixed-point loop is run to
        self-consistently determine the interacting band shift 'B_tilde' (the m=3
        moment correction of Potthoff-Wegner-Nolting, PRB 55, 16132), starting each
        time from its Weiss-field/HF estimate 'B0_tilde'. With the flag False this
        reduces exactly to the KK-IPT-n0 scheme.
        """
        for fl in self.flavors:
            self.calc_G0(mu0_all[fl], fl)

        n0_all = {}
        for fl in self.flavors:
            n0_tmp = float(np.trapezoid(self.G0[fl].N, self.w))
            n0_all[fl] = self._clip_occ(n0_tmp)

        self._calc_IPT_diagram()

        if self.use_potthoff_band_shift:
            for fl in self.flavors:
                other = self._other_flavor(fl)
                self.B0_tilde[fl] = self._calc_band_shift_weiss(
                    fl, n0_fl=n0_all[fl], n0_bar=n0_all[other]
                )

            # Each trial state restarts the inner loop from the Weiss-field estimate,
            # so that the first iteration reduces exactly to the KK coefficient.
            self.B_tilde = dict(self.B0_tilde)

            maxiter = max(1, int(self.band_shift_inner_maxiter))
            self.band_shift_error = 0.0
            self.band_shift_iterations = 0

            # Trial states probed by the outer root-finder can carry an occupation
            # pathologically close to 0 or 1, which blows up the n(1-n) denominator
            # in beta/B_tilde. Bound the inner iterates against a physically motivated
            # scale (onsite energy and U) and, if that bound is ever exceeded, fall
            # back to the Weiss-field estimate B0_tilde instead of propagating a
            # runaway value into the self-energy for this trial state.
            energy_scale = max(abs(self.impurity_onsite_e[fl]) for fl in self.flavors) + abs(self.U) + 1.0
            divergence_bound = 50.0 * energy_scale

            for it in range(maxiter):
                for fl in self.flavors:
                    other = self._other_flavor(fl)
                    self.calc_SE(
                        mu0=mu0_all[fl],
                        n0_bar=n0_all[other],
                        n_bar=n_all[other],
                        fl=fl,
                        recompute_ipt=False,
                    )
                for fl in self.flavors:
                    self.calc_GF(mu0_all[fl], fl)

                B_new = {
                    fl: self._calc_band_shift_interacting(fl, n_fl=n_all[fl])
                    for fl in self.flavors
                }

                if any(
                    not np.isfinite(B_new[fl]) or abs(B_new[fl]) > divergence_bound
                    for fl in self.flavors
                ):
                    self.B_tilde = dict(self.B0_tilde)
                    self.band_shift_diverged_count += 1
                    break

                self.band_shift_error = max(
                    abs(B_new[fl] - self.B_tilde[fl]) for fl in self.flavors
                )
                self.band_shift_iterations = it + 1

                self.B_tilde = {
                    fl: (1.0 - self.band_shift_mixing) * self.B_tilde[fl]
                    + self.band_shift_mixing * B_new[fl]
                    for fl in self.flavors
                }

                if self.band_shift_error < self.band_shift_tol:
                    break

        # Final SE/GF build using the (converged, or plain KK-IPT-n0) coefficients.
        for fl in self.flavors:
            other = self._other_flavor(fl)
            self.calc_SE(
                mu0=mu0_all[fl],
                n0_bar=n0_all[other],
                n_bar=n_all[other],
                fl=fl,
                recompute_ipt=False,
            )

        for fl in self.flavors:
            self.calc_GF(mu0_all[fl], fl)

        return n0_all

    def residual_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the residual of the full four-dimensional vector
        with the solution x = (mu0_up, n_up, mu0_dn, n_dn).

        Residuals are returned in the same flavor order:
        [n - n_trial, n - n0] for each flavor (up, dn).
        """
        mu0_all, n_all = self._unpack_x(x)
        n0_all = self._build_state(mu0_all, n_all)

        res = []
        for fl in self.flavors:
            n_trial = float(np.trapezoid(self.GF[fl].N, self.w))
            res.extend([
                n_all[fl] - n_trial,
                n_all[fl] - n0_all[fl],
            ])

        return np.array(res, dtype=float)

    def calc_occupations(self) -> None:
        """
        Computes the particle number of the impurity GF and Weiss field G0
        via the generalized distribution function and the double occupation.

        n_double uses the exact Keldysh Galitskii-Migdal expression for the
        double occupancy (general, valid in and out of equilibrium):

            n_d = -i/(2*pi*U) * integral dw [Sigma^R(w) G^<(w) + Sigma^<(w) G^A(w)]

        with X^< = X^K/2 - i*Im(X^R) and G^A = (G^R)^*, Sigma^A = (Sigma^R)^*.
        """
        for fl in self.flavors:
            self.n_occ[fl] = np.trapezoid(self.GF[fl].N, self.w)
            self.n0_occ[fl] = np.trapezoid(self.G0[fl].N, self.w)

        fl0 = self.flavors[0]
        GF_R = self.GF[fl0].R
        GF_A = GF_R.conj()
        GF_lt = self.GF[fl0].K / 2 - 1j * np.imag(GF_R)
        SE_R = self.SE[fl0].R
        SE_lt = self.SE[fl0].K / 2 - 1j * np.imag(SE_R)

        integrand = SE_R * GF_lt + SE_lt * GF_A
        self.n_double = np.real(-1j / (2 * np.pi * self.U) * np.trapezoid(integrand, self.w))

    def _restart_strategies(self, x0_initial: np.ndarray) -> list:
        """
        Ordered fallback starting points, tried in turn when the first solve fails.

        In practice the failures come from the root-finder wandering into
        occupations pathologically close to 0 or 1, where the n(1-n) denominators
        in the IPT coefficients (and in B_tilde) blow up. The ladder therefore
        first spreads the starting occupations out, then offers a Weiss field
        whose pole is aligned with the Hartree-shifted impurity level, then
        switches algorithm to Levenberg-Marquardt, and finally jitters around the
        original guess. The jitter is seeded, so the whole sequence is
        reproducible.
        """
        fl_a, fl_b = self.flavors
        zero_mu = {fl: 0.0 for fl in self.flavors}
        half = {fl: 0.5 for fl in self.flavors}
        _, n_initial = self._unpack_x(x0_initial)

        def level_matched(n_all: dict) -> dict:
            """
            Auxiliary potential that puts the Weiss-field pole on the
            Hartree-shifted impurity level. The Weiss field is
            G0_s = 1/(w + mu0_s - Delta_s), so its pole sits at w = -mu0_s,
            while the interacting level sits at eps_s + U n_sbar. Matching the
            two gives mu0_s = -(eps_s + U n_sbar) -- note the OPPOSITE spin's
            occupation, which is what the Hartree term carries.
            """
            return {
                fl: -(self.impurity_onsite_e[fl] + self.U * n_all[self._other_flavor(fl)])
                for fl in self.flavors
            }

        polarized_a = {fl_a: 0.75, fl_b: 0.25}
        polarized_b = {fl_a: 0.25, fl_b: 0.75}

        candidates = [
            ("cold start (mu0=0, n=1/2)", self._pack_x(zero_mu, half), "hybr"),
            # Keeps the incoming occupations but repairs mu0, for the common case
            # of a guess whose n is plausible and whose auxiliary potential is not.
            ("level-matched to the guess occupations",
             self._pack_x(level_matched(n_initial), n_initial), "hybr"),
            # Same construction at half filling; deduplicated against the cold
            # start whenever eps = -U/2 makes the two coincide.
            ("level-matched at half filling", self._pack_x(level_matched(half), half), "hybr"),
            ("polarized towards " + fl_a,
             self._pack_x(level_matched(polarized_a), polarized_a), "hybr"),
            ("polarized towards " + fl_b,
             self._pack_x(level_matched(polarized_b), polarized_b), "hybr"),
            ("Levenberg-Marquardt from the initial guess", np.array(x0_initial, float), "lm"),
        ]

        rng = np.random.default_rng(0)
        mu0_0, n_0 = self._unpack_x(x0_initial)
        for k in range(4):
            mu0_j = {fl: mu0_0[fl] + float(rng.normal(scale=0.5)) for fl in self.flavors}
            n_j = {
                fl: float(np.clip(n_0[fl] + rng.normal(scale=0.15), 0.05, 0.95))
                for fl in self.flavors
            }
            candidates.append((f"jittered restart {k + 1}", self._pack_x(mu0_j, n_j), "hybr"))

        # Drop duplicates, so no restart is wasted repeating a starting point
        # that has already been tried. This matters in practice: at the
        # particle-hole symmetric onsite energy the level-matched guess reduces
        # to mu0 = 0 and would otherwise duplicate the cold start exactly.
        # Duplication is only a problem within the same algorithm -- the same x
        # is worth revisiting with Levenberg-Marquardt.
        selected, seen = [], {"hybr": [np.asarray(x0_initial, float)]}
        for label, x, method in candidates:
            x = np.asarray(x, float)
            if any(np.allclose(x, prev, atol=1e-12) for prev in seen.get(method, [])):
                continue
            seen.setdefault(method, []).append(x)
            selected.append((label, x, method))
        return selected

    def solve(self, *, maxfev: int = 1000, res_tol: float = 1e-8):
        """
        Solves the coupled four-dimensional nonlinear set of equations
        to find the vector (mu0_up, n_up, mu0_dn, n_dn).

        If the root solve fails, up to 'max_restarts' further starting points are
        tried (see '_restart_strategies'). Whatever happens, the state left on the
        solver is rebuilt from the attempt with the smallest residual, not merely
        the last one attempted. If nothing converges, the failure is reported and
        -- unless 'on_convergence_failure' is set to 'warn' -- a ConvergenceError
        is raised, so that a non-solution cannot silently propagate into an outer
        self-consistency loop.
        """
        start_t = time()

        if len(self.flavors) != 2:
            raise ValueError("This implementation assumes exactly 2 flavors.")

        has_guess = (
            hasattr(self, "x_guess")
            and isinstance(self.x_guess, dict)
            and all(fl in self.x_guess for fl in self.flavors)
        )

        if self.use_potthoff_band_shift and not has_guess:
            # A cold start (mu0=0, n=0.5) can push the outer root-finder into trial
            # states with occupations pathologically close to 0 or 1, which makes the
            # inner B_tilde fixed-point loop diverge (see 'divergence_bound' in
            # '_build_state'). The plain KK-IPT-n0 scheme converges reliably from a
            # cold start, so use it to seed a much better initial guess.
            print(
                "No initial guess available: pre-solving the plain KK-IPT-n0 scheme "
                "to warm-start the Potthoff band-shift correction."
            )
            self.use_potthoff_band_shift = False
            try:
                self.solve(maxfev=maxfev, res_tol=res_tol)
            finally:
                # Restore the flag even if the pre-solve raised, so the solver is
                # not silently left in plain KK-IPT-n0 mode.
                self.use_potthoff_band_shift = True
            has_guess = True

        # Default initial guesses
        mu0_guess = {fl: 0.0 for fl in self.flavors}
        n_guess = {fl: 0.5 for fl in self.flavors}

        # Reuse existing guess if present
        if has_guess:
            for fl in self.flavors:
                mu0_guess[fl] = float(self.x_guess[fl][0])
                n_guess[fl] = self._clip_occ(float(self.x_guess[fl][1]))

        x0 = self._pack_x(mu0_guess, n_guess)

        attempts = [("initial guess", x0, "hybr")]
        attempts += self._restart_strategies(x0)[: max(0, int(self.max_restarts))]

        best_label, best_sol, best_res = None, None, np.inf
        for attempt, (label, x_start, method) in enumerate(attempts, start=1):
            self.band_shift_diverged_count = 0
            options = {"maxfev": maxfev} if method == "hybr" else {"maxiter": maxfev}

            sol = root(self.residual_vector, x_start, method=method,
                       tol=res_tol, options=options)
            residual = float(np.linalg.norm(sol.fun))

            # Do NOT trust sol.success on its own. Levenberg-Marquardt in
            # particular reports success once it stops making progress, which it
            # will happily do at a point that is not a root at all: at U=10 with a
            # strongly polarized start it returns success with |residual| ~ 1e-2,
            # against ~1e-12 for the genuine solution. The only safe acceptance
            # criterion is the residual itself.
            accepted = residual <= self.residual_tol

            if accepted or residual < best_res:
                best_label, best_sol, best_res = label, sol, residual

            self.solve_attempts = attempt
            if accepted:
                if attempt > 1:
                    print(f"Recovered on attempt {attempt}/{len(attempts)} "
                          f"using the '{label}' restart.")
                break

            note = "" if not sol.success else " (the algorithm claimed success)"
            print(
                f"WARNING: solve attempt {attempt}/{len(attempts)} "
                f"('{label}') did not converge (|residual| = {residual:.3e} > "
                f"{self.residual_tol:.1e}){note}: {sol.message.strip()}"
            )

        sol = best_sol
        self.solve_strategy = best_label
        self.solve_residual = best_res
        # Report convergence by the residual criterion rather than by the
        # algorithm's own flag, so that a truthy sol.success always means
        # "this really is a root of the coupled system".
        sol.scipy_success = bool(sol.success)
        sol.success = bool(best_res <= self.residual_tol)
        mu0_all, n_all = self._unpack_x(sol.x)

        # Store in the old dict-of-arrays format for compatibility
        self.x_guess = {
            fl: np.array([mu0_all[fl], n_all[fl]], dtype=float)
            for fl in self.flavors
        }
        self.root_solution = sol

        # Final consistent build from the converged solution
        self._build_state(mu0_all, n_all)
        self.calc_occupations()

        print(f"Coupled root solve: success={sol.success}, status={sol.status}")
        print(f"Message: {sol.message}")
        print(f"Residual norm: {np.linalg.norm(sol.fun):.6e}")
        print(f"Time to solve the impurity problem: {time() - start_t:.3f} s")
        print(
            f"Impurity occupations: "
            f"n_{self.flavors[0]} = {self.n_occ[self.flavors[0]]:.6f}, "
            f"n_{self.flavors[1]} = {self.n_occ[self.flavors[1]]:.6f}, "
            f"n_double = {self.n_double:.6f}"
        )

        if self.use_potthoff_band_shift:
            print(
                f"Potthoff m=3 band shift: "
                f"B_tilde = {self.B_tilde}, B0_tilde = {self.B0_tilde}, "
                f"inner error = {self.band_shift_error:.3e}, "
                f"inner iterations = {self.band_shift_iterations}"
            )
            if self.band_shift_diverged_count > 0:
                print(
                    f"WARNING: the inner band-shift fixed-point loop diverged and fell "
                    f"back to the Weiss-field estimate B0_tilde in "
                    f"{self.band_shift_diverged_count} trial state(s) probed by the "
                    f"root-finder. Check 'sol.success' and 'band_shift_error' before "
                    f"trusting the final result."
                )

        if not sol.success:
            failure = (
                f"ERROR: the coupled root solve did not converge after "
                f"{self.solve_attempts} attempt(s) (1 initial + "
                f"{self.solve_attempts - 1} restart(s), 'max_restarts' = "
                f"{self.max_restarts}). Best |residual| = {best_res:.6e}, from the "
                f"'{best_label}' start, against the acceptance tolerance "
                f"'residual_tol' = {self.residual_tol:.1e}. scipy reports: "
                f"{sol.message.strip()} "
                "The Green's function, self-energy and occupations now stored on "
                "this solver belong to that non-converged point: they are NOT a "
                "solution of the impurity problem and must not be fed into an "
                "outer self-consistency loop."
            )
            print(failure)
            if self.on_convergence_failure == "raise":
                raise ConvergenceError(failure)

        return sol
