"""
Error-coordinate coefficients (mu_xi, beta^H, beta^D, sigma_xi).

Two independent code paths are provided:

* ``coefficients_generic`` evaluates the trace formulas (7)-(10) of the
  manuscript directly on a density matrix.  This is the path used by the
  closed-loop simulation; it works for any system.

* ``coefficients_closed_form`` returns the Bloch-coordinate expressions of
  Lemmas V.1-V.2 (qubit, qutrit).  It exists solely as a test oracle:
  ``tests/test_coefficients.py`` asserts the two agree to machine precision,
  which verifies the algebra of Section V.
"""
from __future__ import annotations
import numpy as np
from .systems import System


# --------------------------------------------------------------------------
# Lindblad / innovation superoperators, eq. (3)
# --------------------------------------------------------------------------
def lindblad(Lop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Dissipator  D[L] rho = L rho L^dag - (1/2){L^dag L, rho}."""
    Ld = Lop.conj().T
    return Lop @ rho @ Ld - 0.5 * (Ld @ Lop @ rho + rho @ Ld @ Lop)


def innovation(Lop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Measurement superoperator  H[L] rho = L rho + rho L^dag
       - Tr(L rho + rho L^dag) rho."""
    A = Lop @ rho + rho @ Lop.conj().T
    return A - np.trace(A) * rho


# --------------------------------------------------------------------------
# Generic coefficients (trace formulas)
# --------------------------------------------------------------------------
def coefficients_generic(sys: System, rho: np.ndarray) -> dict:
    """Return mu, betaH (len m), betaD (len p), sigma at state ``rho``.

    mu_xi    = -Tr( Pi (-i[H0,rho] + D[L]rho) )                  (7)
    betaH_i  =  i Tr( [rho,Pi] Hc_i )                            (8)
    betaD_j  = -Tr( Pi D[Lc_j] rho )                             (9)
    sigma_xi = -sqrt(eta) Tr( Pi H[L] rho )                      (10)
    """
    Pi, H0, L = sys.Pi, sys.H0, sys.L
    comm_H0 = H0 @ rho - rho @ H0
    mu = -np.trace(Pi @ (-1j * comm_H0 + lindblad(L, rho)))

    comm_rPi = rho @ Pi - Pi @ rho
    betaH = np.array([1j * np.trace(comm_rPi @ Hc) for Hc in sys.Hc])

    betaD = np.array([-np.trace(Pi @ lindblad(Lc, rho)) for Lc in sys.Lc])

    sigma = -np.sqrt(sys.eta) * np.trace(Pi @ innovation(L, rho))

    out = {
        "mu": _real(mu),
        "betaH": _real(betaH),
        "betaD": _real(betaD),
        "sigma": _real(sigma),
    }
    return out


def _real(z, tol: float = 1e-9):
    """Coefficients are real (Proposition II.1); strip negligible Im part."""
    arr = np.asarray(z)
    if np.max(np.abs(arr.imag)) > tol:
        raise ValueError(f"unexpected imaginary part {np.max(np.abs(arr.imag)):.2e}")
    return arr.real if arr.ndim else float(arr.real)


# --------------------------------------------------------------------------
# Bloch-vector conversions, eq. (42)
# --------------------------------------------------------------------------
def to_bloch(sys: System, rho: np.ndarray) -> np.ndarray:
    """x_k = 2 Tr(T_k rho)."""
    return np.array([2.0 * np.trace(T @ rho).real for T in sys.gens])


def from_bloch(sys: System, x: np.ndarray) -> np.ndarray:
    """rho = I/N + sum_k x_k T_k."""
    rho = np.eye(sys.N, dtype=complex) / sys.N
    for xk, T in zip(x, sys.gens):
        rho = rho + xk * T
    return rho


def infidelity(sys: System, rho: np.ndarray) -> float:
    """xi = 1 - Tr(Pi rho)."""
    return float(1.0 - np.trace(sys.Pi @ rho).real)


# --------------------------------------------------------------------------
# Closed-form Bloch coefficients -- TEST ORACLES (Lemmas V.1, V.2)
# --------------------------------------------------------------------------
def coefficients_closed_form(sys: System, rho: np.ndarray) -> dict:
    """Lemma V.1 / V.2 closed forms, for qubit and qutrit only.

    Used by tests to verify the algebra of Section V against the generic
    trace evaluation.  Coherences are addressed through explicit operators
    rather than Bloch-vector indices, so the result is independent of the
    generator ordering.  The Bell case (Lemma V.4) is convention-sensitive
    and is verified structurally instead -- see test_coefficients.py.
    """
    xi = infidelity(sys, rho)
    if sys.name == "qubit":
        from .systems import SX, SY, SZ
        Omega = _drive_amp(sys.Hc[0])
        Gamma_m = _meas_rate_qubit(sys.L)
        kappa = _kappa(sys.Lc[0])
        x3 = float(np.trace(SZ @ rho).real)          # sigma_z component
        return {
            "mu": 0.0,
            "betaH": np.array([-Omega * float(np.trace(SY @ rho).real)]),
            "betaD": np.array([-kappa * xi]),
            "sigma": -np.sqrt(sys.eta * Gamma_m) * (1.0 - x3 ** 2),
        }
    if sys.name == "qutrit":
        Omega = _drive_amp(sys.Hc[0])
        # lambda_2: the |0>-|1> antisymmetric coherence operator
        lam2 = np.zeros((3, 3), dtype=complex)
        lam2[0, 1] = -1j; lam2[1, 0] = 1j
        P1 = float(rho[1, 1].real)
        P2 = float(rho[2, 2].real)
        kap = [_kappa(Lc) for Lc in sys.Lc]
        return {
            "mu": 0.0,
            "betaH": np.array([-Omega * float(np.trace(lam2 @ rho).real)]),
            "betaD": np.array([-kap[0] * P1, -kap[1] * P2]),
            "sigma": None,   # degree-2 polynomial; verified via generic path
        }
    raise ValueError(f"no closed form registered for system '{sys.name}'")


# -- small helpers to recover scalar parameters from operator data ----------
def _drive_amp(Hc: np.ndarray) -> float:
    return float(np.max(np.abs(Hc)).real)


def _kappa(Lc: np.ndarray) -> float:
    return float(np.max(np.abs(Lc)) ** 2)


def _meas_rate_qubit(L: np.ndarray) -> float:
    return float(np.abs(L[0, 0]) ** 2)
