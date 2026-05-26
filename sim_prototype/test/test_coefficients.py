"""
Verification of the Section V coefficient algebra.

The generic trace-formula evaluation of (mu, betaH, betaD, sigma) is checked
against the closed-form Bloch expressions of Lemmas V.1-V.2 (qubit, qutrit),
and against the structural claims of Lemmas V.3-V.4 / Theorem V.1 (Bell).

Run directly (``python test_coefficients.py``) or under pytest.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import (qubit, qutrit, bell, ggm_generators,
                        coefficients_generic, coefficients_closed_form,
                        infidelity, from_bloch)
from quantumdbc.systems import check_normalisation, _bell_states


def random_rho(N, rng):
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    M = A @ A.conj().T
    return M / np.trace(M)


def test_generator_normalisation():
    for N in (2, 3, 4):
        assert check_normalisation(ggm_generators(N)), f"N={N}"
    print("  [ok] su(N) generators traceless, Tr(T_jT_k)=delta/2 for N=2,3,4")


def test_qubit_coefficients():
    sys_ = qubit()
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        rho = random_rho(2, rng)
        gen = coefficients_generic(sys_, rho)
        cf = coefficients_closed_form(sys_, rho)
        d = (abs(gen["mu"] - cf["mu"])
             + np.max(np.abs(gen["betaH"] - cf["betaH"]))
             + np.max(np.abs(gen["betaD"] - cf["betaD"]))
             + abs(gen["sigma"] - cf["sigma"]))
        worst = max(worst, d)
    assert worst < 1e-10, f"qubit mismatch {worst:.2e}"
    print(f"  [ok] qubit  Lemma V.1 vs generic: max deviation {worst:.2e}")


def test_qutrit_coefficients():
    sys_ = qutrit()
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        rho = random_rho(3, rng)
        gen = coefficients_generic(sys_, rho)
        cf = coefficients_closed_form(sys_, rho)
        d = (abs(gen["mu"] - cf["mu"])
             + np.max(np.abs(gen["betaH"] - cf["betaH"]))
             + np.max(np.abs(gen["betaD"] - cf["betaD"])))
        worst = max(worst, d)
    assert worst < 1e-10, f"qutrit mismatch {worst:.2e}"
    print(f"  [ok] qutrit Lemma V.2 vs generic: max deviation {worst:.2e}")


def test_bell_structural():
    """Lemma V.3-V.4, Corollary V.2, Theorem V.1 -- structural facts."""
    sys_ = bell()
    rng = np.random.default_rng(3)

    # (a) sigma_xi(Pi) = 0  (A4 / Lemma II.2)
    s_Pi = coefficients_generic(sys_, sys_.Pi)["sigma"]
    assert abs(s_Pi) < 1e-10, f"sigma(Pi)={s_Pi:.2e}"

    # (b) betaH = 0 at |00><00|  (Corollary V.2)
    ket00 = np.zeros(4, dtype=complex); ket00[0] = 1.0
    rho00 = np.outer(ket00, ket00.conj())
    bH00 = coefficients_generic(sys_, rho00)["betaH"]
    assert np.max(np.abs(bH00)) < 1e-10, f"betaH(|00>)={bH00}"

    # (c) betaD_k = -kappa_k P_psi_k  (dissipative gains)
    bells = _bell_states()
    psis = [bells["Phi-"], bells["Psi+"], bells["Psi-"]]
    worst_bD = 0.0
    for _ in range(50):
        rho = random_rho(4, rng)
        bD = coefficients_generic(sys_, rho)["betaD"]
        for k, psi in enumerate(psis):
            P = float((psi.conj() @ rho @ psi).real)
            worst_bD = max(worst_bD, abs(bD[k] + 5.0 * P))   # kappa_k = 5
    assert worst_bD < 1e-10, f"betaD mismatch {worst_bD:.2e}"

    # (d) xi >= 1/2 on separable states  (Theorem V.1)
    min_xi = 1.0
    for _ in range(500):
        rA = random_rho(2, rng); rB = random_rho(2, rng)
        rho = np.kron(rA, rB)
        min_xi = min(min_xi, infidelity(sys_, rho))
    assert min_xi >= 0.5 - 1e-9, f"separable xi min = {min_xi:.4f}"

    print("  [ok] Bell    sigma(Pi)=0, betaH(|00>)=0, betaD_k=-kappa_k P_k,")
    print(f"               separable infidelity >= 1/2 (min observed {min_xi:.4f})")


def test_proposition_II1_real():
    """Proposition II.1: all coefficients are real-valued."""
    rng = np.random.default_rng(7)
    for ctor, N in ((qubit, 2), (qutrit, 3), (bell, 4)):
        sys_ = ctor()
        for _ in range(50):
            coefficients_generic(sys_, random_rho(N, rng))   # raises if Im != 0
    print("  [ok] coefficients real-valued for all three systems")


if __name__ == "__main__":
    print("coefficient verification")
    test_generator_normalisation()
    test_qubit_coefficients()
    test_qutrit_coefficients()
    test_bell_structural()
    test_proposition_II1_real()
    print("all coefficient tests passed")
