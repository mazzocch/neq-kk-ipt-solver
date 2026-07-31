"""
Unit-level tests: Keldysh container, Kramers-Kronig helper, Fermi function,
and basic Solver construction/validation -- no full nonlinear solve involved.
"""
import numpy as np
import pytest

from neq_kk_ipt_solver import Keldysh, KK, fermi, Solver


def test_keldysh_spectrum_and_occupation():
    w = np.linspace(-10, 10, 2001)
    R = 1 / (w + 1j * 0.5)
    K = 2j * np.imag(R) * (1 - 2 * fermi(w, 0.0, 0.1))

    g = Keldysh(R=R, K=K)
    g.calc_spectrum()
    g.calc_distribution()
    g.calc_occupation()

    assert g.A is not None
    assert np.all(g.A >= -1e-12)  # spectral function must be non-negative
    assert g.N is not None


def test_keldysh_missing_R_raises():
    g = Keldysh()
    with pytest.raises(ValueError):
        g.calc_spectrum()


def test_keldysh_inverse_is_involution():
    """
    x.inverse().inverse() must reproduce x exactly: the Keldysh-space matrix
    inverse (R_inv=1/R, K_inv=-K/|R|^2) is a genuine involution, independent
    of any specific physics -- a real correctness check on the formula itself.
    """
    w = np.linspace(-10, 10, 501)
    R = 1 / (w + 1j * 0.7 - 0.3)
    K = 2j * np.imag(R) * (1 - 2 * fermi(w, 0.2, 0.15))

    g = Keldysh(R=R, K=K)
    g_back = g.inverse().inverse()

    np.testing.assert_allclose(g_back.R, g.R, rtol=1e-10)
    np.testing.assert_allclose(g_back.K, g.K, rtol=1e-10)


def test_keldysh_inverse_matches_direct_formula():
    w = np.linspace(-10, 10, 501)
    R = 1 / (w + 1j * 0.7 - 0.3)
    K = 2j * np.imag(R) * (1 - 2 * fermi(w, 0.2, 0.15))
    g = Keldysh(R=R, K=K)

    g_inv = g.inverse()

    np.testing.assert_allclose(g_inv.R, 1 / R, rtol=1e-12)
    np.testing.assert_allclose(g_inv.K, -K / np.abs(R) ** 2, rtol=1e-12)


def test_keldysh_sub_keldysh_is_componentwise():
    a = Keldysh(R=np.array([1.0 + 1j, 2.0]), K=np.array([3.0j, 4.0j]))
    b = Keldysh(R=np.array([0.5, 0.5j]), K=np.array([1.0j, 1.0j]))

    c = a - b

    np.testing.assert_allclose(c.R, a.R - b.R)
    np.testing.assert_allclose(c.K, a.K - b.K)


def test_keldysh_sub_scalar_only_affects_R():
    a = Keldysh(R=np.array([1.0 + 1j, 2.0]), K=np.array([3.0j, 4.0j]))

    c = a - 0.5

    np.testing.assert_allclose(c.R, a.R - 0.5)
    np.testing.assert_allclose(c.K, a.K)


def test_fermi_zero_temperature_is_step_function():
    w = np.array([-1.0, -0.001, 0.001, 1.0])
    f = fermi(w, mu=0.0, T=1e-12)
    np.testing.assert_allclose(f, [1.0, 1.0, 0.0, 0.0])


def test_fermi_high_temperature_symmetric_about_mu():
    w = np.linspace(-5, 5, 101)
    f = fermi(w, mu=0.0, T=2.0)
    np.testing.assert_allclose(f + f[::-1], 1.0, atol=1e-10)


def test_kramers_kronig_recovers_known_lorentzian():
    """
    A single Lorentzian's KK-conjugate real part has a known closed form:
    Im[G] = -gamma/((w-w0)^2+gamma^2)  =>  Re[G] = (w-w0)/((w-w0)^2+gamma^2).
    """
    w = np.linspace(-50, 50, 20001)
    gamma = 0.5
    w0 = 0.3

    im_part = -gamma / ((w - w0) ** 2 + gamma ** 2)
    G_R, _ = KK(w, im_part, padding_region=500)

    expected_re = (w - w0) / ((w - w0) ** 2 + gamma ** 2)

    mask = np.abs(w) < 20  # avoid edge/padding artifacts near the boundary
    np.testing.assert_allclose(G_R.real[mask], expected_re[mask], atol=5e-3)
    np.testing.assert_allclose(G_R.imag[mask], im_part[mask], atol=1e-10)


def _minimal_input(**overrides):
    input_data = {
        "global_parameters": {
            "N_points": 2001,
            "w_max": 20.0,
            "U": 3.0,
            "flavors": ["up", "down"],
        },
        "solver": {
            "static": {},
            "modifiable": {},
            "dynamic": {
                "T": 0.05,
                "T_fict": 0.05,
                "D_l": 10.0,
                "D_r": 10.0,
                "t_l": 1.0,
                "t_r": 1.0,
                "mu": 0.0,
                "V": 0.0,
                "output_dir": "./_test_out",
            },
        },
    }
    input_data.update(overrides)
    return input_data


def test_solver_constructs_and_sets_ph_symmetric_onsite_energy():
    solver = Solver(_minimal_input())
    assert solver.impurity_onsite_e["up"] == pytest.approx(-solver.U / 2)
    assert solver.impurity_onsite_e["down"] == pytest.approx(-solver.U / 2)


def test_solver_rejects_missing_solver_key():
    bad_input = {"global_parameters": {
        "N_points": 101, "w_max": 5.0, "U": 1.0, "flavors": ["up", "down"]
    }}
    with pytest.raises(ValueError):
        Solver(bad_input)
