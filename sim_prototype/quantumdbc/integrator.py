"""
Strong-order-one Milstein propagation of the controlled SME (Section IV-H).

The state is carried as the generalised Bloch vector x in R^(N^2-1):

    dx = F(x; u, gamma) dt + G(x) dW,

    F_k = 2 <T_k, -i[H0 + sum u_i Hc_i, rho] + D[L]rho + sum gamma_j D[Lc_j]rho>
    G_k = 2 sqrt(eta) <T_k, H[L] rho>,            <A,B> = Tr(A^dag B).

The Milstein correction (G . grad) G is evaluated by a central finite
difference of the vector field G along its own direction.
"""
from __future__ import annotations
import numpy as np
from .systems import System
from .coefficients import lindblad, innovation, from_bloch


def _project(sys: System, X: np.ndarray) -> np.ndarray:
    """Coordinates of a Hermitian operator X on the generators: 2 Tr(T_k X)."""
    return np.array([2.0 * np.trace(T @ X).real for T in sys.gens])


def drift(sys: System, x: np.ndarray, u: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Deterministic vector field F(x; u, gamma)."""
    rho = from_bloch(sys, x)
    H = sys.H0.copy()
    for ui, Hc in zip(u, sys.Hc):
        H = H + ui * Hc
    drho = -1j * (H @ rho - rho @ H) + lindblad(sys.L, rho)
    for gj, Lc in zip(gamma, sys.Lc):
        drho = drho + gj * lindblad(Lc, rho)
    return _project(sys, drho)


def diffusion(sys: System, x: np.ndarray) -> np.ndarray:
    """Diffusion vector field G(x)."""
    rho = from_bloch(sys, x)
    return np.sqrt(sys.eta) * _project(sys, innovation(sys.L, rho))


def milstein_correction(sys: System, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(G . grad) G at x, by a central difference along the direction G(x)."""
    G = diffusion(sys, x)
    Gp = diffusion(sys, x + eps * G)
    Gm = diffusion(sys, x - eps * G)
    return (Gp - Gm) / (2.0 * eps)


def milstein_step(sys: System, x: np.ndarray, u: np.ndarray, gamma: np.ndarray,
                  dt: float, dW: float) -> np.ndarray:
    """One Milstein step, eq. (45)."""
    F = drift(sys, x, u, gamma)
    G = diffusion(sys, x)
    GgG = milstein_correction(sys, x)
    return x + F * dt + G * dW + 0.5 * GgG * (dW ** 2 - dt)


def euler_step(sys: System, x: np.ndarray, u: np.ndarray, gamma: np.ndarray,
               dt: float, dW: float) -> np.ndarray:
    """Euler-Maruyama step (strong order 1/2); for convergence comparison."""
    F = drift(sys, x, u, gamma)
    G = diffusion(sys, x)
    return x + F * dt + G * dW


def project_physical(sys: System, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Project the reconstructed state onto {rho >= 0, Tr rho = 1}.

    Finite-step integration of the Bloch-vector SDE can leave the physical
    state space; the SME vector field is only contractive on it.  This
    projection clips negative eigenvalues and renormalises the trace, a
    standard positivity-preserving safeguard for SME integrators.  Returns
    the projected Bloch vector and the magnitude of the negative-eigenvalue
    excursion that was removed (0 if the state was already physical).
    """
    from .coefficients import from_bloch, to_bloch
    rho = from_bloch(sys, x)
    rho = 0.5 * (rho + rho.conj().T)
    w, V = np.linalg.eigh(rho)
    neg = float(-w.min()) if w.min() < 0 else 0.0
    if neg == 0.0 and abs(np.trace(rho).real - 1.0) < 1e-12:
        return x, 0.0
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        rho_p = np.eye(sys.N, dtype=complex) / sys.N
    else:
        rho_p = (V * (w / s)) @ V.conj().T
    return to_bloch(sys, rho_p), neg
