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

    Notes
    -----
    A non-converged solve leaves `G`, `Sigma` and the occupations at
    whatever point the root-finder stopped at. Those are not a solution of
    the impurity problem, and feeding them into an outer loop (e.g. DMFT)
    silently poisons every subsequent iteration. Failing loudly is
    therefore the default; set ``on_convergence_failure`` to ``"warn"`` to
    get the old behaviour of returning the non-converged result instead.
    """


class Solver:
    """
    Nonequilibrium Kajueter-Kotliar IPT impurity solver.

    Construct from a JSON-schema-checked input dictionary, call
    :meth:`solve`, then read off :attr:`n_occ`, :attr:`n_double`,
    :attr:`GF`, :attr:`SE`. See the package README for the full physics
    background and the input-parameter reference.

    Parameters
    ----------
    input_data : dict
        Solver input, with top-level keys ``"global_parameters"`` (grid,
        `U`, flavor names) and ``"solver"`` (``"static"``, ``"modifiable"``,
        ``"dynamic"`` sub-sections -- solver flags, the onsite energy, and
        the hybridization/bath specification, respectively). Validated
        against ``schemas/solver.schema.json`` and
        ``schemas/global_parameters.schema.json``.

    Attributes
    ----------
    w : numpy.ndarray
        Frequency grid, ``linspace(-w_max, w_max, N_points)``.
    flavors : list[str]
        The two spin-flavor labels (e.g. ``["up", "down"]``).
    G0 : dict[str, Keldysh]
        Weiss field, per flavor.
    GF : dict[str, Keldysh]
        Interacting impurity Green's function, per flavor.
    SE : dict[str, Keldysh]
        Self-energy, per flavor.
    Delta : dict[str, Keldysh]
        Hybridization function, per flavor.
    n_occ : dict[str, float]
        Converged occupations, per flavor.
    n_double : float
        Double occupancy, averaged over both flavors (see
        :meth:`calc_occupations`).
    n_double_per_flavor : dict[str, float]
        The two flavor-resolved double-occupancy estimates `n_double` is
        averaged from.
    solve_residual : float
        Residual norm of the accepted solution (see :meth:`solve`).

    Raises
    ------
    ValueError
        If ``input_data`` has no ``"solver"`` key, or
        ``on_convergence_failure`` is not ``"raise"``/``"warn"``.
    """

    def __init__(self, input_data: dict) -> None:
        if "solver" not in input_data.keys():
            raise ValueError("ERROR: No input parameters for the solver in input JSON file!")

        set_global_parameters(self, input_data)

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_data = input_data

        # Microseconds are kept deliberately. Truncating to whole seconds meant
        # two solves started within the same second produced the same output
        # filename, and the second silently overwrote the first -- easy to hit
        # in a fast outer loop that stores every iteration.
        now = datetime.datetime.now()
        formatted_now = now.isoformat()

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
        self.n_double_per_flavor = {}
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
        Apply ``solver.static``/``modifiable``/``dynamic`` and derive `w`, `impurity_onsite_e`.
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
        """Clip an occupation into ``(0, 1)``, away from the exact endpoints where 1/n(1-n) diverges."""
        return float(np.clip(n, 1e-8, 1 - 1e-8))

    def _other_flavor(self, fl: str) -> str:
        """Return the flavor label that is not `fl` (this solver assumes exactly 2 flavors)."""
        if len(self.flavors) != 2:
            raise ValueError("This implementation assumes exactly 2 flavors.")
        return self.flavors[1] if fl == self.flavors[0] else self.flavors[0]

    def _lesser(self, G: Keldysh) -> np.ndarray:
        r"""Lesser component, :math:`G^< = G^K/2 - i\,\mathrm{Im}\,G^R`."""
        return G.K / 2.0 - 1j * np.imag(G.R)

    def _greater(self, G: Keldysh) -> np.ndarray:
        r"""Greater component, :math:`G^> = G^K/2 + i\,\mathrm{Im}\,G^R`."""
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
        """Pack per-flavor ``(mu0, n)`` dicts into the flat vector the root-finder solves for."""
        vals = []
        for fl in self.flavors:
            vals.extend([float(mu0_dict[fl]), self._clip_occ(float(n_dict[fl]))])
        return np.array(vals, dtype=float)

    def _unpack_x(self, x: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        """Inverse of :meth:`_pack_x`: unpack the flat solver vector back into per-flavor ``(mu0, n)`` dicts."""
        if len(x) != 2 * len(self.flavors):
            raise ValueError("Input vector has wrong size for the number of flavors.")

        mu0_dict = {}
        n_dict = {}

        for i, fl in enumerate(self.flavors):
            mu0_dict[fl] = float(x[2 * i])
            n_dict[fl] = self._clip_occ(float(x[2 * i + 1]))

        return mu0_dict, n_dict

    def store_output(self) -> None:
        """
        Write the full solution to a JSON file under ``output_dir``.

        No-op if ``store`` is `False`. The written file contains, per
        flavor, the Green's function and self-energy arrays, occupations,
        interpolation coefficients and band-shift diagnostics; plus
        convergence provenance (``converged``, ``residual_norm``,
        ``solve_attempts``, ``solve_strategy``), the resolved input
        parameters, and a timestamp/git-commit metadata block. Filename is
        ``solver_output_<ISO-timestamp-with-microseconds>.json``.
        """
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
                "n_double": self.n_double_per_flavor[fl],
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
        """Set the ``spin_sym`` flag after construction.

        Parameters
        ----------
        spin_sym : bool
            Whether to enforce spin symmetry (both flavors share the same
            onsite energy).
        """
        self.spin_sym = spin_sym

    def set_onsite_energies_external(self, impurity_onsite_e: dict[str, float]) -> None:
        """
        Override the per-flavor onsite energy from outside, e.g. from a DMFT self-consistency loop.

        Parameters
        ----------
        impurity_onsite_e : dict[str, float]
            New onsite energy, per flavor.
        """
        self.impurity_onsite_e = impurity_onsite_e

    def set_Delta_external(self, Delta: Keldysh) -> None:
        """
        Override the hybridization function from outside, e.g. from a DMFT self-consistency loop.

        Bypasses :meth:`_set_Delta` entirely, so any of the four
        ``solver.dynamic`` hybridization specifications can be replaced by
        an externally computed one between solves.

        Parameters
        ----------
        Delta : dict[str, Keldysh]
            New hybridization function, per flavor.
        """
        self.Delta = Delta

    def set_output_external(self, directory: str, filename: str) -> None:
        """
        Override the output directory and filename set by ``solver.dynamic``.

        Parameters
        ----------
        directory : str
            New output directory.
        filename : str
            New output filename.
        """
        self.output_dir = directory
        self.filename = filename

    def _set_Delta(self) -> None:
        """
        Build the hybridization function :attr:`Delta` from ``solver.dynamic``.

        Tried in order of precedence, reading from attributes already set by
        :meth:`_load_config`:

        1. explicit ``Delta_R_im``/``Delta_K_im`` arrays (fully general, per
           flavor);
        2. a semi-elliptic ("semicircular") band of half-bandwidth
           ``Delta_D`` centered on ``Delta_center`` (both required, as for
           the Lorentzian);
        3. a Lorentzian of center ``Delta_center`` and width ``Delta_gamma``;
        4. the box-shaped-DOS fallback built from ``T_fict``, ``D_l``, ``D_r``,
           ``t_l``, ``t_r``.

        For 2. and 3., each shape parameter may be a single number
        (spin-symmetric/degenerate hybridization, shared by both flavors) or
        a per-flavor dict (spin-dependent hybridization; see
        :meth:`_as_flavor_dict`).
        """
        try:
            box_shaped_dos = False
            _ = self.Delta_R_im
        except Exception:
            box_shaped_dos = True

        semicircular_hyb = False
        lorentzian_hyb = False
        if box_shaped_dos:
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

        if box_shaped_dos:
            print("No hybridization has been given. A box-shaped DOS with fictitious inverse temperature and fixed bandwidth is being used.")

        for fl in self.flavors:
            if box_shaped_dos:
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
        r"""
        Construct the Weiss field :attr:`G0` for one flavor from the hybridization.

        Kajueter-Kotliar modified-IPT Weiss field [Phys. Rev. Lett. 77, 131
        (1996)]:

        .. math::

            \mathcal{G}_{0,\sigma}^{-1}(\omega) = \omega + \mu_{0,\sigma} - \Delta_\sigma(\omega)

        written as the literal Dyson equation
        :math:`\mathcal{G}_{0,\sigma} = (\mathcal{G}_{0,\sigma}^{-1})^{-1}`,
        with :math:`\Delta_\sigma` playing the role of a self-energy here.

        Parameters
        ----------
        mu0 : float
            Auxiliary chemical potential for flavor `fl`. Caller's
            responsibility to pass the value that actually corresponds to
            `fl`.
        fl : str
            Which flavor to build the Weiss field for.
        """
        G0_inv = Keldysh(R=self.w + mu0, K=0) - self.Delta[fl]
        self.G0[fl] = G0_inv.inverse()

        self.G0[fl].calc_spectrum()
        self.G0[fl].calc_distribution()
        self.G0[fl].calc_occupation()

    def _calc_IPT_diagram(self, enforce_anti_herm: bool = True) -> None:
        r"""
        Compute the bare second-order (Kajueter-Kotliar) bubble diagram, :attr:`IPT_diag`.

        .. math::

            \tilde\Sigma_\sigma(\tau) = U^2\,\mathcal{G}_{0,\sigma}(\tau)\,
            \mathcal{G}_{0,\bar\sigma}(\tau)\,\mathcal{G}_{0,\bar\sigma}(-\tau)

        for both flavors, from the current :attr:`G0`. This is the raw
        diagram that :meth:`calc_SE` feeds into the KK-IPT interpolation
        Ansatz -- not the self-energy itself.

        Parameters
        ----------
        enforce_anti_herm : bool, default=True
            If `True`, discard the (numerically spurious) real part of the
            greater/lesser components before reconstructing the retarded
            part via Kramers-Kronig. Measured directly (several
            representative cases, see
            ``test_ipt_diagram_anti_hermiticity_enforcement_discards_only_noise``
            in ``tests/test_solver_basics.py``): that real part sits at
            double-precision machine epsilon relative to the imaginary part
            kept, and disabling this entirely reproduces the same solved
            occupations -- so `False` is equally correct, and `True` stays
            the default only for the simplicity of a guaranteed-clean
            invariant, not because it is numerically necessary.
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

            # Not numerically necessary: the discarded real part is machine
            # epsilon relative to the imaginary part kept (measured directly,
            # see test_ipt_diagram_anti_hermiticity_enforcement_discards_only_noise),
            # and disabling this reproduces the same solved occupations.
            # Kept as the default anyway, purely so downstream code can rely
            # on diag_gtr/diag_lss being exactly purely imaginary without
            # having to reason about whether that holds.
            if enforce_anti_herm:
                diag_gtr = 1j * diag_gtr.imag
                diag_lss = 1j * diag_lss.imag

            diag_R_imag = 0.5 * (diag_gtr - diag_lss).imag
            diag_K = diag_gtr + diag_lss

            diag_R, _ = KK(self.w, diag_R_imag)

            self.IPT_diag[fl].R = diag_R
            self.IPT_diag[fl].K = diag_K

    def calc_SE(self, mu0: float, n0_bar: float, n_bar: float, fl: str, recompute_ipt: bool = True) -> None:
        r"""
        Compute the self-energy :attr:`SE` for one flavor via the generalized KK-IPT Ansatz.

        .. math::

            \Sigma_\sigma^R(\omega) = U n_{\bar\sigma}
            + \frac{\alpha_\sigma \tilde\Sigma_\sigma^R(\omega)}
            {1 - \beta_\sigma \tilde\Sigma_\sigma^R(\omega)}

        with the interpolation coefficients

        .. math::

            \alpha_\sigma = \frac{n_{\bar\sigma}(1-n_{\bar\sigma})}{n_{0,\bar\sigma}(1-n_{0,\bar\sigma})},
            \qquad
            \beta_\sigma = \frac{(1-n_{\bar\sigma})U + \epsilon_{f,\sigma} + \mu_{0,\sigma}}
            {n_{0,\bar\sigma}(1-n_{0,\bar\sigma}) U^2}

        fixed by exact atomic-limit matching, and :math:`\tilde\Sigma_\sigma`
        the bare bubble diagram from :meth:`_calc_IPT_diagram`. The Keldysh
        component :attr:`SE`\ ``.K`` follows the same coefficients (see the
        package README for the full expression).

        The subscript "bar" denotes the quantities of the opposite spin
        species. When no subscript is present it is implied that the
        quantity corresponding to the current flavor `fl` has been passed
        correctly.

        Parameters
        ----------
        mu0 : float
            Auxiliary chemical potential for flavor `fl`.
        n0_bar : float
            Weiss-field occupation of the *opposite* flavor.
        n_bar : float
            True occupation of the *opposite* flavor.
        fl : str
            Which flavor to compute the self-energy for.
        recompute_ipt : bool, default=True
            If `True`, call :meth:`_calc_IPT_diagram` first. Pass `False`
            when the diagram is already current (e.g. inside the Potthoff
            inner loop of :meth:`_build_state`, where `G0` does not change
            between iterations).

        Notes
        -----
        If ``use_potthoff_band_shift`` is `True`, :math:`\beta_\sigma`
        (the coefficient above) is additionally corrected by the
        nonequilibrium analogue of the Potthoff-Wegner-Nolting [Phys. Rev. B
        55, 16132 (1997)] :math:`m=3` moment band-shift term:

        .. math::

            \beta_\sigma = \frac{(1-n_{\bar\sigma})U + \epsilon_{f,\sigma} + \mu_{0,\sigma}
            + \big(\tilde B_{\bar\sigma} - \tilde B_{0,\bar\sigma}\big)}
            {n_{0,\bar\sigma}(1-n_{0,\bar\sigma}) U^2}

        i.e. ``B_tilde[other] - B0_tilde[other]`` is added to the plain
        numerator above -- a genuinely new additive term, not a substitution
        of anything already present (:math:`\epsilon_{f,\bar\sigma}` does not
        otherwise appear in :math:`\beta_\sigma`). It vanishes, and the plain
        coefficient is recovered exactly, whenever
        :math:`\tilde B_{\bar\sigma} = \tilde B_{0,\bar\sigma}` -- in
        particular at the start of the inner fixed-point loop described in
        :meth:`_build_state`, and always in the paramagnetic case. See
        :doc:`../theory/potthoff_band_shift` for the full derivation and why
        this restores the exact :math:`M_3` spectral moment.
        :math:`\tilde B_{\bar\sigma}`, :math:`\tilde B_{0,\bar\sigma}` must
        have been (re)computed beforehand, e.g. in :meth:`_build_state`.
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
        r"""
        Compute the interacting Green's function :attr:`GF` for one flavor.

        .. math::

            G_\sigma = \left(\mathcal{G}_{0,\sigma}^{-1} - \mathrm{shift}_\sigma - \Sigma_\sigma\right)^{-1}

        the literal Dyson equation, where
        :math:`\mathrm{shift}_\sigma = \mu_{0,\sigma} + \epsilon_{f,\sigma}`
        is the auxiliary-potential correction that must be subtracted back
        out of :math:`\mathcal{G}_{0,\sigma}`'s own inverse (see
        :meth:`calc_G0`).

        Parameters
        ----------
        mu0 : float
            Auxiliary chemical potential for flavor `fl`.
        fl : str
            Which flavor to compute the Green's function for. Requires
            :attr:`G0` and :attr:`SE` for this flavor to already be current.
        """
        shift = mu0 + self.impurity_onsite_e[fl]
        GF_inv = self.G0[fl].inverse() - shift - self.SE[fl]
        self.GF[fl] = GF_inv.inverse()

        self.GF[fl].calc_spectrum()
        self.GF[fl].calc_distribution()
        self.GF[fl].calc_occupation()

    def _calc_band_shift_weiss(self, fl: str, *, n0_fl: float, n0_bar: float) -> float:
        r"""
        Weiss/Hartree-Fock-level band shift :math:`\tilde B_{0,\sigma}`.

        The nonequilibrium Keldysh analogue of Potthoff-Wegner-Nolting
        [Phys. Rev. B 55, 16132 (1997)] Eq. (13)/(29):

        .. math::

            \tilde B_{0,\sigma} = \epsilon_{f,\sigma} + \frac{2n_{0,\bar\sigma}-1}
            {n_{0,\sigma}(1-n_{0,\sigma})}\,(-i)\int\frac{d\omega}{2\pi}
            \big[\mathcal{G}_{0,\sigma}\Delta_\sigma\big]^<(\omega)

        with the lesser component of the product expanded via the Langreth rules,

        .. math::

            \big[\mathcal{G}_{0,\sigma}\Delta_\sigma\big]^<(\omega) =
            \mathcal{G}_{0,\sigma}^R(\omega)\,\Delta_\sigma^<(\omega)
            + \mathcal{G}_{0,\sigma}^<(\omega)\,\Delta_\sigma^A(\omega)

        Parameters
        ----------
        fl : str
            Flavor to compute :math:`\tilde B_{0,\sigma}` for.
        n0_fl : float
            Weiss-field occupation of `fl`.
        n0_bar : float
            Weiss-field occupation of the opposite flavor.

        Returns
        -------
        float
            :math:`\tilde B_{0,\sigma}`.

        Notes
        -----
        A functional of :attr:`G0` and :attr:`Delta` alone -- unlike its
        interacting counterpart :attr:`B_tilde` (see
        :meth:`_calc_band_shift_interacting`), it needs no self-energy or
        interacting Green's function, so it is cheap to (re)compute at the
        start of every trial state in :meth:`_build_state`, before the inner
        fixed-point loop begins. It seeds that loop: :math:`\tilde B_\sigma`
        starts out equal to :math:`\tilde B_{0,\sigma}`, at which point the
        correction below vanishes and :meth:`calc_SE` reduces exactly to the
        plain (uncorrected) coefficients.

        It combines with the same flavor's interacting counterpart as
        :math:`\tilde B_\sigma - \tilde B_{0,\sigma}`; that difference is
        what actually enters the self-energy of the *opposite* flavor
        :math:`\bar\sigma` via :meth:`calc_SE` (see Eq. below in
        :meth:`_calc_band_shift_interacting`).
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
        r"""
        Interacting band shift :math:`\tilde B_\sigma`.

        The nonequilibrium Keldysh analogue of Potthoff-Wegner-Nolting
        [Phys. Rev. B 55, 16132 (1997)] Eq. (44) with
        :math:`Q_\sigma = 2\Sigma_\sigma/U - 1`:

        .. math::

            \tilde B_\sigma = \epsilon_{f,\sigma} + \frac{1}{n_\sigma(1-n_\sigma)}\,
            (-i)\int\frac{d\omega}{2\pi}\big[Q_\sigma G_\sigma \Delta_\sigma\big]^<(\omega)

        with the lesser component of the triple product expanded via the
        Langreth rules,

        .. math::

            \big[Q_\sigma G_\sigma \Delta_\sigma\big]^< =
            Q_\sigma^R G_\sigma^R \Delta_\sigma^< + Q_\sigma^R G_\sigma^< \Delta_\sigma^A
            + Q_\sigma^< G_\sigma^A \Delta_\sigma^A

        where :math:`Q_\sigma^R = 2\Sigma_\sigma^R/U - 1` and
        :math:`Q_\sigma^< = 2\Sigma_\sigma^</U`.

        Parameters
        ----------
        fl : str
            Flavor to compute :math:`\tilde B_\sigma` for. Requires
            :attr:`SE`\ ``[fl]`` and :attr:`GF`\ ``[fl]`` to already
            correspond to the current trial state.
        n_fl : float
            True occupation of `fl`.

        Returns
        -------
        float
            :math:`\tilde B_\sigma`.

        Raises
        ------
        ValueError
            If `U` is (numerically) zero.
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
        Build `G0`, the IPT diagram, `SE`, `GF` consistently for all flavors from trial `(mu0, n)`.

        Parameters
        ----------
        mu0_all : dict[str, float]
            Trial auxiliary chemical potential, per flavor.
        n_all : dict[str, float]
            Trial occupation, per flavor.

        Returns
        -------
        dict[str, float]
            The resulting Weiss-field occupation, per flavor (``n0_all``).

        Notes
        -----
        If ``use_potthoff_band_shift`` is `True`, an inner fixed-point loop
        is run to self-consistently determine the interacting band shift
        :attr:`B_tilde` (the :math:`m=3` moment correction of
        Potthoff-Wegner-Nolting [Phys. Rev. B 55, 16132 (1997)]), starting
        each time from its Weiss-field/HF estimate :attr:`B0_tilde`. With
        the flag `False` this reduces exactly to the KK-IPT-n0 scheme.
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
        Residual of the coupled 4D self-consistency, for use with `scipy.optimize.root`.

        Parameters
        ----------
        x : numpy.ndarray
            Packed trial vector ``(mu0_up, n_up, mu0_down, n_down)`` (see
            :meth:`_pack_x`).

        Returns
        -------
        numpy.ndarray
            Residuals in the same flavor order: ``[n - n_trial, n - n0]``
            for each flavor, where `n_trial` is the occupation integrated
            from the just-built `GF` and `n0` from `G0`. Zero at the
            self-consistent solution.
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
        r"""
        Compute :attr:`n_occ`, :attr:`n0_occ` and :attr:`n_double`.

        :attr:`n_double` uses the exact Keldysh Galitskii-Migdal expression
        for the double occupancy (general, valid in and out of
        equilibrium), evaluated per flavor and averaged over the spin
        species:

        .. math::

            n_d = -\frac{i}{2\pi U}\cdot\frac{1}{2}\sum_\sigma\int d\omega\,
            \big[\Sigma^R_\sigma(\omega) G^<_\sigma(\omega) + \Sigma^<_\sigma(\omega) G^A_\sigma(\omega)\big]

        with :math:`X^< = X^K/2 - i\,\mathrm{Im}\,X^R` and
        :math:`X^A = (X^R)^*`.

        Notes
        -----
        This is an exact identity, so the two flavor-resolved estimates
        must coincide for the true interacting solution -- but within an
        approximate self-energy scheme they need not agree exactly once the
        solution is spin-polarized. The individual values are kept in
        :attr:`n_double_per_flavor` as a diagnostic: a large spread between
        them is a direct measure of how badly the approximation violates
        this exact flavor-independence.
        """
        for fl in self.flavors:
            self.n_occ[fl] = np.trapezoid(self.GF[fl].N, self.w)
            self.n0_occ[fl] = np.trapezoid(self.G0[fl].N, self.w)

        self.n_double_per_flavor = {}

        for fl in self.flavors:

            GF_R = self.GF[fl].R
            GF_A = GF_R.conj()
            GF_lt = self.GF[fl].K / 2 - 1j * np.imag(GF_R)
            SE_R = self.SE[fl].R
            SE_lt = self.SE[fl].K / 2 - 1j * np.imag(SE_R)

            integrand = SE_R * GF_lt + SE_lt * GF_A
            self.n_double_per_flavor[fl] = np.real(-1j / (2 * np.pi * self.U) * np.trapezoid(integrand, self.w))

        self.n_double = np.mean(list(self.n_double_per_flavor.values()))

    def _restart_strategies(self, x0_initial: np.ndarray) -> list:
        """
        Ordered fallback starting points, tried in turn when the first solve fails.

        Parameters
        ----------
        x0_initial : numpy.ndarray
            The initial packed guess that just failed, used to seed several
            of the fallback candidates (e.g. the jittered restarts).

        Returns
        -------
        list[tuple[str, numpy.ndarray, str]]
            ``(label, x_start, method)`` triples, duplicates against
            `x0_initial` removed, in the order they should be tried.

        Notes
        -----
        In practice the failures come from the root-finder wandering into
        occupations pathologically close to 0 or 1, where the :math:`n(1-n)`
        denominators in the IPT coefficients (and in `B_tilde`) blow up. The
        ladder therefore first spreads the starting occupations out, then
        offers a Weiss field whose pole is aligned with the Hartree-shifted
        impurity level, then switches algorithm to Levenberg-Marquardt, and
        finally jitters around the original guess. The jitter is seeded, so
        the whole sequence is reproducible.
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
        Solve the coupled four-dimensional self-consistency for ``(mu0_up, n_up, mu0_down, n_down)``.

        Parameters
        ----------
        maxfev : int, default=1000
            Maximum number of function evaluations per solve attempt,
            passed to `scipy.optimize.root`.
        res_tol : float, default=1e-8
            Tolerance passed to `scipy.optimize.root`. Acceptance of the
            result is decided separately, by ``residual_tol`` against the
            actual residual norm (see Notes).

        Returns
        -------
        scipy.optimize.OptimizeResult
            The accepted (or best found) root-finder result. ``sol.success``
            reflects the residual-norm acceptance criterion, not the
            optimizer's own flag (preserved separately as
            ``sol.scipy_success``). Also populates :attr:`n_occ`,
            :attr:`n_double`, :attr:`GF`, :attr:`SE`, :attr:`G0`,
            :attr:`solve_attempts`, :attr:`solve_strategy` and
            :attr:`solve_residual`.

        Raises
        ------
        ConvergenceError
            If no attempt converges and ``on_convergence_failure`` is
            ``"raise"`` (the default).

        Notes
        -----
        If the root solve fails, up to ``max_restarts`` further starting
        points are tried (see :meth:`_restart_strategies`). Whatever
        happens, the state left on the solver is rebuilt from the attempt
        with the smallest residual, not merely the last one attempted. If
        nothing converges, the failure is reported and -- unless
        ``on_convergence_failure`` is set to ``"warn"`` -- a
        :class:`ConvergenceError` is raised, so that a non-solution cannot
        silently propagate into an outer self-consistency loop.
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
