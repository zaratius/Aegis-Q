"""
Operator data for the three Quantum Dynamic Barrier Control instances.

Conventions
-----------
* Generators {T_k}, k = 1 .. N^2-1, are the traceless Hermitian generators
  of su(N), normalised by  <T_j, T_k> = Tr(T_j^dag T_k) = (1/2) delta_jk.
  This is the convention of eq. (42) of the manuscript, used uniformly for
  every N.  For N = 2, 3 it coincides with sigma_k/2 and lambda_k/2.
* A density operator is rho = I/N + sum_k x_k T_k, hence x_k = 2 Tr(T_k rho).

The qutrit and Bell operators are the *corrected* data of the Section V
rewrite: the qutrit uses two engineered dissipators that each pump an
excited level directly to the ground state, and the Bell measurement is a
joint-parity readout L = sqrt(Gamma_m) sigma_z (x) sigma_z (QND w.r.t.
|Phi+>).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# --------------------------------------------------------------------------
# Pauli matrices
# --------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)        # |0><1|, lowering to ground


# --------------------------------------------------------------------------
# Generalised Gell-Mann basis: traceless Hermitian, Tr(T_j T_k) = delta_jk/2
# --------------------------------------------------------------------------
def ggm_generators(N: int) -> list[np.ndarray]:
    """Return the N^2-1 generators of su(N), normalised to Tr(T_jT_k)=delta/2."""
    gens: list[np.ndarray] = []
    # symmetric off-diagonal
    for j in range(N):
        for k in range(j + 1, N):
            T = np.zeros((N, N), dtype=complex)
            T[j, k] = T[k, j] = 0.5
            gens.append(T)
    # antisymmetric off-diagonal
    for j in range(N):
        for k in range(j + 1, N):
            T = np.zeros((N, N), dtype=complex)
            T[j, k] = -0.5j
            T[k, j] = +0.5j
            gens.append(T)
    # diagonal
    for l in range(1, N):
        d = np.zeros(N)
        d[:l] = 1.0
        d[l] = -float(l)
        T = np.diag(d).astype(complex) / np.sqrt(2.0 * l * (l + 1))
        gens.append(T)
    return gens


def check_normalisation(gens: list[np.ndarray], tol: float = 1e-12) -> bool:
    """Verify Tr(T_jT_k) = delta_jk/2 and tracelessness."""
    n = len(gens)
    for j in range(n):
        if abs(np.trace(gens[j])) > tol:
            return False
        for k in range(n):
            expected = 0.5 if j == k else 0.0
            if abs(np.trace(gens[j] @ gens[k]) - expected) > tol:
                return False
    return True


# --------------------------------------------------------------------------
# System container
# --------------------------------------------------------------------------
@dataclass
class System:
    """Operator data for one instance.

    Attributes
    ----------
    name        : label
    N           : Hilbert-space dimension
    Pi          : target projector  |psi><psi|
    H0          : drift Hamiltonian
    Hc          : list of coherent control Hamiltonians  (length m)
    L           : measurement (Lindblad) operator
    Lc          : list of engineered dissipator jump operators  (length p)
    eta         : homodyne detection efficiency, in (0, 1]
    gens        : su(N) generators (Tr T_jT_k = delta/2)
    """
    name: str
    N: int
    Pi: np.ndarray
    H0: np.ndarray
    Hc: list[np.ndarray]
    L: np.ndarray
    Lc: list[np.ndarray]
    eta: float
    gens: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not self.gens:
            self.gens = ggm_generators(self.N)

    @property
    def m(self) -> int:
        return len(self.Hc)

    @property
    def p(self) -> int:
        return len(self.Lc)


# --------------------------------------------------------------------------
# Instance constructors
# --------------------------------------------------------------------------
def qubit(omega_q=5.0, Omega=20.0, Gamma_m=1.0, kappa=5.0, eta=0.6) -> System:
    """Qubit ground-state stabilisation, eq. (50) of the rewrite."""
    Pi = np.array([[1, 0], [0, 0]], dtype=complex)
    H0 = 0.5 * omega_q * SZ
    Hc = [Omega * SX]
    L = np.sqrt(Gamma_m) * SZ
    Lc = [np.sqrt(kappa) * SM]
    return System("qubit", 2, Pi, H0, Hc, L, Lc, eta)


def qutrit(omega01=5.0, alpha_anh=-0.3, Omega=20.0, Gamma_m=1.0,
           kappa1=5.0, kappa2=5.0, eta=0.6) -> System:
    """Qutrit leakage suppression, eq. (53) of the rewrite (corrected:
    single coherent gate drive, two direct-to-ground dissipators)."""
    N = 3
    ket = lambda i: np.eye(N, dtype=complex)[i]
    op = lambda i, j: np.outer(ket(i), ket(j).conj())
    Pi = op(0, 0)
    H0 = np.diag([0.0, omega01, 2 * omega01 + alpha_anh]).astype(complex)
    lam1 = op(0, 1) + op(1, 0)                       # |0><->|1| drive
    Hc = [Omega * lam1]
    lam8 = np.diag([1.0, 1.0, -2.0]).astype(complex) / np.sqrt(3.0)
    L = np.sqrt(Gamma_m) * lam8                      # QND dispersive readout
    Lc = [np.sqrt(kappa1) * op(0, 1),                # |1> -> |0>
          np.sqrt(kappa2) * op(0, 2)]                # |2> -> |0>
    return System("qutrit", N, Pi, H0, Hc, L, Lc, eta)


def _bell_states() -> dict[str, np.ndarray]:
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket01 = np.array([0, 1, 0, 0], dtype=complex)
    ket10 = np.array([0, 0, 1, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)
    s = 1 / np.sqrt(2)
    return {
        "Phi+": s * (ket00 + ket11),
        "Phi-": s * (ket00 - ket11),
        "Psi+": s * (ket01 + ket10),
        "Psi-": s * (ket01 - ket10),
    }


def bell(omega1=5.0, omega2=5.0, Omega1=20.0, Omega2=20.0, Gamma_m=1.0,
         kappa=(5.0, 5.0, 5.0), eta=0.6) -> System:
    """Bell-state stabilisation, eq. (55) of the rewrite (corrected:
    joint-parity measurement L = sqrt(Gamma_m) sigma_z (x) sigma_z)."""
    bells = _bell_states()
    phi_p = bells["Phi+"]
    Pi = np.outer(phi_p, phi_p.conj())
    kron = np.kron
    H0 = 0.5 * omega1 * kron(SZ, I2) + 0.5 * omega2 * kron(I2, SZ)
    Hc = [Omega1 * kron(SX, I2), Omega2 * kron(I2, SX)]
    L = np.sqrt(Gamma_m) * kron(SZ, SZ)              # joint-parity, QND wrt Phi+
    Lc = []
    for k, name in enumerate(["Phi-", "Psi+", "Psi-"]):
        psi = bells[name]
        Lc.append(np.sqrt(kappa[k]) * np.outer(phi_p, psi.conj()))
    return System("bell", 4, Pi, H0, Hc, L, Lc, eta)
