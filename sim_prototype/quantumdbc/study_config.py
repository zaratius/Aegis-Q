"""Single source of truth for the closed-loop study geometry.

Every runner imports from here so the funnel / buffer can never disagree
across figures again. Before this file, run_e1_acc used (0.55, 0.08, s_b=0.04)
while run_breach used (0.70, 0.12, s_b=0.25): the confinement figure and the
breach figure described DIFFERENT systems, which makes the feasibility-vs-
confinement story impossible to write honestly.

Geometry constraints (both must hold, or the theory breaks):

  1. s_b < eps_T
     The barrier floors the margin at s_b. If s_b > eps_T, then once the
     funnel contracts below s_b the design shell Omega(t) = {xi <= eps(t)-s_b}
     becomes EMPTY (eps(T)-s_b < 0), and you cannot confine to an empty shell.
     The old Table II values (s_b=0.25, eps_T=0.12) violated this: the shell
     was empty for the back half of the horizon. That is the most likely
     single cause of the persistent breaches.

  2. s_b "large enough" to cap noise pressure
     The log-barrier curvature is zeta_V = 1/s, capped at 1/s_b on the shell,
     so noise pressure <= sigma^2 / (2 s_b). A pressure-capping buffer needs
     s_b large; a tight terminal precision needs s_b small. Both can only be
     satisfied if eps_T is not too tight. eps_T=0.30, s_b=0.20 satisfies both.

If you WANT the tight eps_T=0.12 funnel, that is a legitimate choice -- but
then the honest paper reports that it is infeasible near the target
(Remark III.3), and you may not also claim a.s. confinement on it. Pick one.
"""

import numpy as np
from quantumdbc import ExpFunnel

# --- Qubit study geometry (E1 confinement, breach refinement, robustness) ---
QUBIT_FUNNEL = ExpFunnel(eps0=0.70, eps_T=0.30, T=4.0)   # was eps_T=0.12
QUBIT_SB     = 0.28                                       # was 0.25 (> eps_T!)
QUBIT_LAM    = 0.5
QUBIT_RHO0   = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)

# Shared QP weights (identical across the qubit studies)
QUBIT_WEIGHTS = dict(wr=50.0, c=20.0, wgamma=1.0, wdelta=1e3, regularized=True)

# Hard guard 1: refuse a geometry that empties the shell at t = T.
assert QUBIT_SB < QUBIT_FUNNEL.eps_T, (
    f"buffer s_b={QUBIT_SB} must be < terminal funnel eps_T="
    f"{QUBIT_FUNNEL.eps_T}, else the design shell is empty at t=T"
)

# Hard guard 2: Problem 1 precondition eps(0) > xi(rho0) + s_b.
_xi0 = 1.0 - float(QUBIT_RHO0[0, 0].real)        # qubit error: xi = 1 - <0|rho|0>
assert QUBIT_FUNNEL.eps0 > _xi0 + QUBIT_SB, (
    f"Problem 1 precondition violated: eps0={QUBIT_FUNNEL.eps0} must exceed "
    f"xi(rho0)={_xi0:.3f} + s_b={QUBIT_SB}"
)