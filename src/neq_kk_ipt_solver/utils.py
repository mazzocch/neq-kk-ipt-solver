"""
Support utilities for the Solver class: the Keldysh container, the Fermi
function, a Kramers-Kronig (Hilbert-transform) helper, JSON-schema-backed
global-parameter loading, and a small git-provenance helper for output
metadata.

Trimmed from the parent project's utils.py down to exactly
what neq_kk_ipt_solver.solver.Solver uses.

Author: Tommaso Maria Mazzocchi
"""
import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from scipy.signal import hilbert


@dataclass
class Keldysh:
    """
    Container for a Green's-function-like object in the Keldysh formalism:
    retarded (R) and Keldysh (K) components, plus the derived spectral
    function (A), (generalized) distribution function (F), and occupation density (N).
    """
    R: np.ndarray | None = None
    K: np.ndarray | None = None
    A: np.ndarray | None = None
    F: np.ndarray | None = None
    N: np.ndarray | None = None

    def calc_spectrum(self):
        if self.R is not None:
            self.A = -np.imag(self.R) / np.pi
        else:
            raise ValueError("Attribute 'R' in Keldysh class is 'None': cannot compute spectrum.")

    def calc_distribution(self):
        """
        Generalized distribution function F = [1 - Im K / (2 Im R)] / 2.

        F is undefined wherever the spectral weight vanishes: Im R -> 0 makes
        the ratio 0/0 or x/0, which on a wide frequency grid happens over the
        whole region beyond the bands. Those points carry no spectral weight,
        so F has no meaning there and is set to NaN rather than +-inf -- and,
        more to the point, computing it no longer floods stderr with
        divide-by-zero RuntimeWarnings on a perfectly healthy solve.

        F is diagnostic only. The occupation density N is built from A and K
        directly (see calc_occupation), so nothing in the solve depends on it.
        """
        if self.R is None or self.K is None:
            raise ValueError("Attributes 'R' or 'K' in Keldysh class are 'None': cannot compute distribution function.")

        denominator = 2 * np.imag(self.R)
        ratio = np.divide(
            np.imag(self.K), denominator,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=denominator != 0,
        )
        self.F = 0.5 * (1 - ratio)

    def calc_occupation(self):
        if self.A is not None and self.F is not None:
            self.N = self.A / 2 + np.imag(self.K) / (4 * np.pi)
        else:
            raise ValueError("Attributes 'R' or 'K' in Keldysh class are 'None': cannot compute occupation function.")

    def inverse(self) -> "Keldysh":
        """
        Keldysh-space matrix inverse of this (local, scalar) retarded/Keldysh
        pair. Writing the object as the upper-triangular Keldysh matrix
        [[R, K], [0, A]] with A = R^*, its inverse works out to

            R_inv = 1 / R
            K_inv = -K / |R|^2

        This is a genuine involution: x.inverse().inverse() reproduces x
        exactly (see test_keldysh_inverse_is_involution).

        Writing a Dyson equation as G = (G0.inverse() - shift - Sigma).inverse()
        is the literal textbook G = (G0^-1 - Sigma)^-1, e.g. in calc_G0/calc_GF.
        """
        R_inv = 1 / self.R
        K_inv = -self.K / np.abs(self.R) ** 2
        return Keldysh(R=R_inv, K=K_inv)

    def __sub__(self, other) -> "Keldysh":
        """
        Keldysh - Keldysh: componentwise (R-R, K-K).
        Keldysh - scalar/array: subtracted from R only, K untouched -- a
        real energy shift (e.g. a chemical potential) has no Keldysh
        component of its own.
        """
        if isinstance(other, Keldysh):
            return Keldysh(R=self.R - other.R, K=self.K - other.K)
        return Keldysh(R=self.R - other, K=self.K)


def fermi(w: np.ndarray, mu: float, T: float) -> np.ndarray:
    """
    Fermi-Dirac distribution, guarded against exponential overflow/underflow.
    """
    if T < 1e-10:
        return np.heaviside(mu - w, 0.5)

    energy = w - mu
    pos_exp = energy / T > 35
    neg_exp = energy / T < -35

    f = np.zeros_like(energy)

    f[pos_exp] = np.exp(-energy[pos_exp] / T)
    f[neg_exp] = 1.0
    f[~(pos_exp | neg_exp)] = 1 / (np.exp(energy[~(pos_exp | neg_exp)] / T) + 1)

    return f


def KK(w: np.ndarray, GF_R_im: np.ndarray, padding_region=500, same=True) -> tuple[np.ndarray, int]:
    """
    Calculates the real part of a retarded Green's function from its imaginary
    part via the Kramers-Kronig relation (implemented as a Hilbert transform).

    Parameters
    ----------
    w : np.ndarray
        Frequency grid.
    GF_R_im : np.ndarray
        Imaginary part of the retarded function.
    padding_region : float
        Extra frequency range (same units as w) zero-padded on both sides
        before the Hilbert transform, to reduce edge/wrap-around artifacts.
    same : bool
        If True (default), returns an array trimmed back to the original
        (unpadded) length.
    """
    delta_w = w[1] - w[0]
    points_padding = int(padding_region // delta_w)
    points_data = len(GF_R_im)

    GF_R_im = np.concatenate((np.zeros(points_padding), GF_R_im, np.zeros(points_padding)))
    GF_R = 1j * hilbert(GF_R_im)

    if same:
        GF_R = GF_R[points_padding: points_padding + points_data]

    return GF_R, points_padding


def get_git_commit_hash() -> str:
    """Best-effort git commit hash of the current repo, for output provenance metadata."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        print(f"Error obtaining Git commit hash: {e}")
        return "unknown"


def set_global_parameters(class_instance, input_file: dict) -> None:
    """
    Validates input_file["global_parameters"] against global_parameters.schema.json
    and sets each entry as an attribute on class_instance, plus the derived
    frequency grid class_instance.w = linspace(-w_max, w_max, N_points).
    """
    if "global_parameters" not in input_file.keys():
        raise ValueError("ERROR: The global input parameters are not provided!")

    if "strict" in input_file["global_parameters"].keys():
        class_instance.strict = input_file["global_parameters"]["strict"]
    else:
        class_instance.strict = False

    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(dir_path, "schemas", "global_parameters.schema.json"), "r") as file:
        json_schema = json.load(file)

    try:
        validate(instance=input_file["global_parameters"], schema=json_schema)
    except ValidationError as e:
        if class_instance.strict:
            raise ValueError("\n\nERROR: Global parameter input JSON data is INVALID!!\n\n")
        else:
            print("\n\nWARNING: Global parameter input JSON data is INVALID!!\n\n")
            print("Validation error:", e.message)

    for key, value in input_file["global_parameters"].items():
        setattr(class_instance, key, value)

    class_instance.w = np.linspace(
        -class_instance.w_max, class_instance.w_max, class_instance.N_points
    )
