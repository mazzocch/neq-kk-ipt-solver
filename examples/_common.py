"""
Shared helpers for the examples.

Kept separate so each example script stays about the physics it is
demonstrating rather than about plotting. Not an example itself, hence the
leading underscore.
"""
import os

import numpy as np

COLOR = {"up": "#c1121f", "down": "#0353a4"}
SYMBOL = {"up": r"\uparrow", "down": r"\downarrow"}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def spin_resolved(solver, tol: float = 1e-6) -> bool:
    """True when the two flavors are actually different."""
    return abs(float(np.real(solver.n_occ["up"] - solver.n_occ["down"]))) > tol


def spectral_window(solver, keep: float = 0.99, pad: float = 1.15) -> tuple:
    """
    Symmetric plotting window holding `keep` of the spectral weight, so each
    example frames itself instead of needing a hand-tuned range.

    Framing on cumulative weight rather than on where A exceeds some fraction
    of its maximum matters here: a broad bath leaves small but very long tails,
    and an amplitude criterion follows them out to the edge of the grid,
    squashing the Hubbard bands and the quasiparticle peak into an
    indistinguishable hump.
    """
    w = solver.w
    edge = 0.0
    for fl in solver.flavors:
        A = -np.imag(solver.GF[fl].R) / np.pi
        cumulative = np.cumsum(A) * (w[1] - w[0])
        cumulative /= cumulative[-1]
        lo = w[np.searchsorted(cumulative, 0.5 * (1 - keep))]
        hi = w[np.searchsorted(cumulative, 1 - 0.5 * (1 - keep))]
        edge = max(edge, abs(lo), abs(hi))
    return -pad * edge, pad * edge


def report(solver, sol) -> None:
    """Console summary. The same numbers go onto the figure."""
    print()
    print("  converged            : "
          f"{sol.success}   (|residual| = {solver.solve_residual:.2e}, "
          f"{solver.solve_attempts} attempt(s))")
    for fl in solver.flavors:
        print(f"  n_{fl:<18}: {float(np.real(solver.n_occ[fl])):.6f}")
    print(f"  double occupancy     : {solver.n_double:.6f}")
    print(f"  sum of occupations   : "
          f"{sum(float(np.real(solver.n_occ[fl])) for fl in solver.flavors):.6f}")


def plot_solution(solver, sol, title: str, subtitle: str, filename: str) -> str:
    """
    Spectral function and retarded self-energy, with the occupations and the
    double occupancy annotated on the figure so it stands on its own.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved = spin_resolved(solver)
    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    for fl in solver.flavors:
        label = rf"$\sigma={SYMBOL[fl]}$" if resolved else "both spins"
        colour = COLOR[fl] if resolved else "#3d348b"
        ax_a.plot(solver.w, -np.imag(solver.GF[fl].R) / np.pi,
                  color=colour, lw=1.8, label=label)
        ax_s.plot(solver.w, np.imag(solver.SE[fl].R), color=colour, lw=1.8, label=label)
        if not resolved:
            break  # the two curves coincide; drawing one is honest and legible

    lo, hi = spectral_window(solver)
    for ax in (ax_a, ax_s):
        ax.axvline(0.0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$\omega$")
        ax.grid(alpha=0.18, lw=0.6)
        ax.legend(frameon=False, fontsize=9.5)

    ax_a.set_ylabel(r"$A_\sigma(\omega)$")
    ax_a.set_title(r"spectral function", fontsize=10.5, loc="left")
    ax_s.set_ylabel(r"$\mathrm{Im}\,\Sigma^{R}_\sigma(\omega)$")
    ax_s.set_title(r"retarded self-energy (imaginary part)", fontsize=10.5, loc="left")

    lines = [rf"$n_\uparrow = {float(np.real(solver.n_occ['up'])):.4f}$",
             rf"$n_\downarrow = {float(np.real(solver.n_occ['down'])):.4f}$",
             rf"$n_{{\rm d}} = {solver.n_double:.4f}$"]
    ax_a.text(0.025, 0.97, "\n".join(lines), transform=ax_a.transAxes,
              va="top", ha="left", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9))
    ax_a.set_ylim(0.0, ax_a.get_ylim()[1] * 1.30)

    fig.suptitle(title, fontsize=12.5, y=0.985)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=9.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.875))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figure               : {os.path.relpath(path, os.getcwd())}")
    return path


def run(build_input, title: str, subtitle: str, filename: str):
    """Solve, report, plot. The body every example shares."""
    from neq_kk_ipt_solver import Solver

    solver = Solver(build_input(OUTPUT_DIR))
    sol = solver.solve()
    report(solver, sol)
    plot_solution(solver, sol, title, subtitle, filename)
    return solver, sol
