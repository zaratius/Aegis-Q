import numpy as np
from quantumdbc import ExpFunnel
from quantumdbc.systems import bell

QUBIT_FUNNEL  = ExpFunnel(eps0=0.70, eps_T=0.0001, T=4.0)
QUBIT_THETA_B = 0.20        
QUBIT_LAM     = 0.5         
QUBIT_RHO0    = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)

QUBIT_WEIGHTS = dict(c=20.0, wgamma=1.0, wdelta=1e4, gmax=3.90,
                     regularized=True)

_lhs_q = (8.0 * 0.6 * (1.0 - QUBIT_THETA_B)
          * (1.0 - (1.0 - QUBIT_THETA_B) * QUBIT_FUNNEL.eps_T) ** 2
          / QUBIT_THETA_B)
assert _lhs_q <= 5.0 * QUBIT_WEIGHTS["gmax"], (
    f"qubit shell edge not defensible: {_lhs_q:.2f} > kappa*gmax = "
    f"{5.0 * QUBIT_WEIGHTS['gmax']:.2f}"
)

assert 0.0 < QUBIT_THETA_B < 1.0, (
    f"theta_b={QUBIT_THETA_B} must lie in (0,1)"
)

_xi0 = 1.0 - float(QUBIT_RHO0[0, 0].real)       
assert _xi0 < (1.0 - QUBIT_THETA_B) * QUBIT_FUNNEL.eps0, (
    f"Problem 1 precondition violated: xi(rho0)={_xi0:.3f} must be below "
    f"(1-theta_b) eps0 = {(1.0 - QUBIT_THETA_B) * QUBIT_FUNNEL.eps0:.3f}"
)


BELL_FUNNEL  = ExpFunnel(eps0=0.85, eps_T=0.0001, T=4.0)
BELL_THETA_B = 0.20
BELL_LAM     = 0.5
BELL_DT      = 2.5e-4
BELL_KAPPA   = 15.0

BELL_WEIGHTS = dict(c=20.0, wgamma=1.0, wdelta=1e3, umax=1.0, gmax=1.30,
                    regularized=True)


def bell_system():
    """The certified-study Bell system: rotating frame (H0 = 0), kappa = 50."""
    return bell(omega1=0.0, omega2=0.0,
                kappa=(BELL_KAPPA, BELL_KAPPA, BELL_KAPPA))


def bell_rho0() -> np.ndarray:

    rho0 = np.zeros((4, 4), dtype=complex)
    rho0[0, 0] = 1.0
    return rho0


# Preparation-error tilt for the bell sim
BELL_THETA_PREP = 0.2          


def bell_rho0_prep(theta: float = BELL_THETA_PREP) -> np.ndarray:
    """rho0 = |psi><psi| with |psi> = (Rx(theta) x I)|00>."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    psi = np.array([c, 0.0, -1j * s, 0.0], dtype=complex)   # c|00> - i s|10>
    return np.outer(psi, psi.conj())


assert 0.0 < BELL_THETA_B < 1.0, (
    f"theta_b={BELL_THETA_B} must lie in (0,1)"
)


_sys_bell = bell_system()
_xi0_bell = 1.0 - float(np.trace(_sys_bell.Pi @ bell_rho0_prep()).real)
assert _xi0_bell < (1.0 - BELL_THETA_B) * BELL_FUNNEL.eps0, (
    f"Bell Problem 1 precondition violated: xi(rho0)={_xi0_bell:.3f} must be "
    f"below (1-theta_b) eps0 = {(1.0 - BELL_THETA_B) * BELL_FUNNEL.eps0:.3f}"
)

_lhs = (8.0 * _sys_bell.eta * 1.0 * (1.0 - BELL_THETA_B)
        * (1.0 - (1.0 - BELL_THETA_B) * BELL_FUNNEL.eps_T) ** 2 / BELL_THETA_B)
assert _lhs <= BELL_KAPPA * BELL_WEIGHTS["gmax"], (
    f"Bell shell edge not defensible: {_lhs:.2f} > kappa*gmax = "
    f"{BELL_KAPPA * BELL_WEIGHTS['gmax']:.2f}"
)
