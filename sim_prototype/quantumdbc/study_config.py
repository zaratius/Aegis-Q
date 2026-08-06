"""Single source of truth for the closed-loop study geometry.

Every runner imports from here so the funnel / buffer can never disagree
across figures again. Before this file, run_e1_acc used (0.55, 0.08, s_b=0.04)
while run_breach used (0.70, 0.12, s_b=0.25): the confinement figure and the
breach figure described DIFFERENT systems, which makes the feasibility-vs-
confinement story impossible to write honestly.

The buffer is now RELATIVE (funnel gauge, revised Section III):

    s_b(t) = theta_b * eps(t),   theta_b in (0, 1).

Geometry facts under the gauge:

  1. The design shell Omega(t) = {xi <= (1-theta_b) eps(t)} is never empty
     for theta_b < 1. The old fixed-buffer failure mode -- s_b > eps_T
     emptying the shell over the back half of the horizon, the likely cause
     of the persistent breaches in the early runs -- cannot occur.

  2. Noise-pressure cap: zeta_V = 1/s <= 1/(theta_b eps(t)), worst at the
     horizon where it reaches 1/(theta_b eps_T). A larger theta_b caps the
     pressure harder (feasibility wants theta_b LARGE).

  3. Doob exit level: the shell sits at the constant barrier level
     log(1/theta_b). A larger theta_b lowers the level and weakens the
     a-priori bound of Corollary III.2 (the certificate wants theta_b SMALL).

The theta_b choice trades 2 against 3; the feasibility sweep (fig_feas) is
the arbiter.

2026-07-23 update (second pass, supersedes the first note): the closed-loop
envelope bound was vacuous (~14) at (lam=0.5, theta_b=0.35, wdelta=1e3,
gmax=1.0). Root cause, from the corrected envelope (quantumdbc.envelope,
variant keeping the -lam V credit and the pointwise margin): the worst shell
point is the zero-coherent-gain ridge (x2=0) at the shell's outer edge near
the horizon, gamma saturated -- there delta* is wdelta-INDEPENDENT, and the
edge is defensible only when

    8 eta Gm (1-theta_b) (1-(1-theta_b) eps_T)^2 / theta_b <= kappa gmax.

At theta_b=0.35 the LHS is 5.78 > kappa*gmax = 5: the funnel was genuinely
infeasible there, and no envelope tightening or wdelta can hide that.

Two configs certify once the edge is fixed (fine grid, corrected envelope):
    (lam=0.5, theta_b=0.35, gmax=1.25, wdelta=1e4) -> bound ~0.53  <- ADOPTED
    (lam=0.1, theta_b=0.45, gmax=1.00, wdelta=1e4) -> bound ~0.70  (first-note
     config; keeps gmax=1 at the cost of a lower Doob level. NOTE: the first
     note's lam=0.1 helped only under the OLD envelope, which failed to credit
     -lam V back; with the corrected envelope lam nets out of the bound
     (0.494 vs 0.505 at lam=0.5 vs 0.1), so lam stays at 0.5 for transient
     shaping.)
"""

import numpy as np
from quantumdbc import ExpFunnel

# --- Qubit study geometry (E1 confinement, breach refinement, robustness) ---
QUBIT_FUNNEL  = ExpFunnel(eps0=0.70, eps_T=0.30, T=4.0)
QUBIT_THETA_B = 0.35        # paper Table I value; edge defensible given gmax=1.25
QUBIT_LAM     = 0.5         # restored (lam nets out of the corrected bound)
QUBIT_RHO0    = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)

# Shared QP weights (identical across the qubit studies).  gmax=1.25 is the
# edge-defensibility fix: kappa*gmax = 6.25 >= 5.78 (see docstring).
QUBIT_WEIGHTS = dict(wr=50.0, c=20.0, wgamma=1.0, wdelta=1e4, gmax=1.25,
                     regularized=True)

# Hard guard 1: the relative buffer must be a proper fraction.
assert 0.0 < QUBIT_THETA_B < 1.0, (
    f"theta_b={QUBIT_THETA_B} must lie in (0,1)"
)

# Hard guard 2: Problem 1 precondition xi(rho0) < (1-theta_b) eps(0).
_xi0 = 1.0 - float(QUBIT_RHO0[0, 0].real)        # qubit error: xi = 1 - <0|rho|0>
assert _xi0 < (1.0 - QUBIT_THETA_B) * QUBIT_FUNNEL.eps0, (
    f"Problem 1 precondition violated: xi(rho0)={_xi0:.3f} must be below "
    f"(1-theta_b) eps0 = {(1.0 - QUBIT_THETA_B) * QUBIT_FUNNEL.eps0:.3f}"
)
