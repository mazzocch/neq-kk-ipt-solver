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
    r"""
    Container for a Green's-function-like object in the Keldysh formalism.

    Holds the retarded (`R`) and Keldysh (`K`) components on a common
    frequency grid, plus three quantities derived from them: the spectral
    function `A`, the generalized distribution function `F`, and the
    occupation density `N`. `Solver.G0`, `Solver.GF`, `Solver.SE` and
    `Solver.Delta` are all dictionaries of `Keldysh` objects, one per spin
    flavor.

    Parameters
    ----------
    R : numpy.ndarray of complex, optional
        Retarded component, :math:`G^R(\omega)`.
    K : numpy.ndarray of complex, optional
        Keldysh component, :math:`G^K(\omega)`.
    A : numpy.ndarray of float, optional
        Spectral function, :math:`A(\omega) = -\mathrm{Im}\,G^R(\omega)/\pi`.
        Populated by :meth:`calc_spectrum`.
    F : numpy.ndarray of float, optional
        Generalized distribution function. Populated by
        :meth:`calc_distribution`.
    N : numpy.ndarray of float, optional
        Occupation density (integrates to the total occupation). Populated
        by :meth:`calc_occupation`.
    """
    R: np.ndarray | None = None
    K: np.ndarray | None = None
    A: np.ndarray | None = None
    F: np.ndarray | None = None
    N: np.ndarray | None = None

    def calc_spectrum(self) -> None:
        r"""
        Populate :attr:`A` from :attr:`R`.

        .. math::

            A(\omega) = -\frac{1}{\pi}\,\mathrm{Im}\,G^R(\omega)

        Raises
        ------
        ValueError
            If :attr:`R` has not been set.
        """
        if self.R is not None:
            self.A = -np.imag(self.R) / np.pi
        else:
            raise ValueError("Attribute 'R' in Keldysh class is 'None': cannot compute spectrum.")

    def calc_distribution(self) -> None:
        r"""
        Populate :attr:`F` from :attr:`R` and :attr:`K`.

        .. math::

            F(\omega) = \frac{1}{2}\left[1 - \frac{\mathrm{Im}\,G^K(\omega)}
            {2\,\mathrm{Im}\,G^R(\omega)}\right]

        Notes
        -----
        `F` is undefined wherever the spectral weight vanishes:
        :math:`\mathrm{Im}\,G^R \to 0` makes the ratio :math:`0/0` or
        :math:`x/0`, which on a wide frequency grid happens over the whole
        region beyond the bands. Those points carry no spectral weight, so
        `F` has no meaning there and is set to `NaN` rather than
        :math:`\pm\infty` -- and, more to the point, computing it no longer
        floods stderr with divide-by-zero `RuntimeWarning`\ s on a perfectly
        healthy solve.

        `F` is diagnostic only. The occupation density :attr:`N` is built
        from `A` and `K` directly (see :meth:`calc_occupation`), so nothing
        in the solve depends on it.

        Raises
        ------
        ValueError
            If :attr:`R` or :attr:`K` has not been set.
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

    def calc_occupation(self) -> None:
        r"""
        Populate :attr:`N` from :attr:`A` and :attr:`K`.

        .. math::

            N(\omega) = \frac{A(\omega)}{2} + \frac{\mathrm{Im}\,G^K(\omega)}{4\pi}

        Raises
        ------
        ValueError
            If :attr:`A` or :attr:`F` has not been set (:meth:`calc_spectrum`
            and :meth:`calc_distribution` must be called first).
        """
        if self.A is not None and self.F is not None:
            self.N = self.A / 2 + np.imag(self.K) / (4 * np.pi)
        else:
            raise ValueError("Attributes 'R' or 'K' in Keldysh class are 'None': cannot compute occupation function.")

    def inverse(self) -> "Keldysh":
        r"""
        Keldysh-space matrix inverse of this (local, scalar) retarded/Keldysh pair.

        Writing the object as the upper-triangular Keldysh matrix
        :math:`\begin{pmatrix}X^{R} & X^{K}\\0 & X^{A}\end{pmatrix}` with
        :math:`X^{A} = (X^{R})^*`, its inverse works out to

        .. math::

            (X^{R})^{-1} = \frac{1}{X^{R}}, \qquad
            (X^{K})^{-1} = -\frac{X^{K}}{|X^{R}|^2}

        Returns
        -------
        Keldysh
            The inverted retarded/Keldysh pair.

        Notes
        -----
        This is a genuine involution: ``x.inverse().inverse()`` reproduces
        `x` exactly (see ``test_keldysh_inverse_is_involution``).

        Writing a Dyson equation as
        ``G = (G0.inverse() - shift - Sigma).inverse()`` is the literal
        textbook :math:`G = (G_0^{-1} - \Sigma)^{-1}`, e.g. in
        :meth:`Solver.calc_G0 <neq_kk_ipt_solver.Solver.calc_G0>` /
        :meth:`Solver.calc_GF <neq_kk_ipt_solver.Solver.calc_GF>`.
        """
        R_inv = 1 / self.R
        K_inv = -self.K / np.abs(self.R) ** 2
        return Keldysh(R=R_inv, K=K_inv)

    def __sub__(self, other) -> "Keldysh":
        """
        Subtract another `Keldysh` object, or a real scalar/array, from this one.

        Parameters
        ----------
        other : Keldysh or scalar or numpy.ndarray
            If a `Keldysh` object: subtracted componentwise (``R-R``,
            ``K-K``). Otherwise: subtracted from `R` only, `K` left
            untouched -- a real energy shift (e.g. a chemical potential) has
            no Keldysh component of its own.

        Returns
        -------
        Keldysh
            The difference.
        """
        if isinstance(other, Keldysh):
            return Keldysh(R=self.R - other.R, K=self.K - other.K)
        return Keldysh(R=self.R - other, K=self.K)


def fermi(w: np.ndarray, mu: float, T: float) -> np.ndarray:
    r"""
    Fermi-Dirac distribution, guarded against exponential overflow/underflow.

    .. math::

        f(\omega) = \frac{1}{e^{(\omega-\mu)/T} + 1}

    Below ``T=1e-10`` this falls back to the exact zero-temperature step
    function instead of evaluating the (numerically unstable) formula above.

    Parameters
    ----------
    w : numpy.ndarray
        Frequencies at which to evaluate the distribution.
    mu : float
        Chemical potential.
    T : float
        Temperature (same units as `w`; not an inverse temperature).

    Returns
    -------
    numpy.ndarray
        :math:`f(\omega)`, same shape as `w`.
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
    r"""
    Calculates the real part of a retarded Green's function from its imaginary
    part via the Kramers-Kronig relation (implemented as a Hilbert transform).

    Parameters
    ----------
    w : numpy.ndarray
        Frequency grid.
    GF_R_im : numpy.ndarray
        Imaginary part of the retarded function.
    padding_region : float, default=500
        Extra frequency range (same units as `w`) zero-padded on both sides
        before the Hilbert transform, to reduce edge/wrap-around artifacts.
    same : bool, default=True
        If `True`, returns an array trimmed back to the original (unpadded)
        length.

    Returns
    -------
    GF_R : numpy.ndarray of complex
        The retarded function, :math:`\mathrm{Re}\,G^R` reconstructed via
        Kramers-Kronig and :math:`\mathrm{Im}\,G^R` equal to the input
        `GF_R_im`.
    points_padding : int
        Number of zero-padding points added on each side, so a caller with
        ``same=False`` knows how to trim the result later.
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
    """
    Best-effort git commit hash of the current repo, for output provenance metadata.

    Returns
    -------
    str
        The current ``HEAD`` commit hash, or ``"unknown"`` if it could not
        be determined (e.g. not a git checkout, or `git` unavailable).
    """
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
    Validate and apply the ``global_parameters`` section of a solver input.

    Validates ``input_file["global_parameters"]`` against
    ``global_parameters.schema.json`` and sets each entry (``N_points``,
    ``w_max``, ``U``, ``flavors``, ...) as an attribute directly on
    `class_instance`, plus the derived frequency grid
    ``class_instance.w = linspace(-w_max, w_max, N_points)``.

    Parameters
    ----------
    class_instance : object
        Typically a :class:`~neq_kk_ipt_solver.Solver` instance; any object
        that attributes can be set on.
    input_file : dict
        The full solver input dictionary; must contain a
        ``"global_parameters"`` key.

    Raises
    ------
    ValueError
        If ``"global_parameters"`` is missing, or (when
        ``global_parameters["strict"]`` is `True`) if it fails schema
        validation. Otherwise a schema failure only prints a warning.
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
