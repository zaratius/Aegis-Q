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

2026-08-06 update (Markovian law): the temporal regularizer w_r and the
carried command u_prev are removed from the QP, matching the revised
manuscript. The paired ablation (M=100, seeds 1000-1099) showed w_r had no
trajectory-level effect at the adopted config (max |Delta(xi/eps)| < 5e-7,
0/100 classification changes) because the qubit runs with the coherent
channel effectively inactive (x2(0)=0 invariant under u=0, and w_u large
near betaH=0). Removing it also removes the history bias from the envelope
sup, which drops the certified bound to essentially the slack-free Doob
floor V0/log(1/theta_b) ~ 0.36 -- see the numbers memo of that date.
"""

import numpy as np
from quantumdbc import ExpFunnel
from quantumdbc.systems import bell

# --- Qubit study geometry (E1 confinement, breach refinement, robustness) ---
#
# 2026-08-08 redesign (user decision): thin buffer theta_b = 0.20 to match
# the certified Bell study's geometry visually and tighten the Doob level.
# The edge-defensibility condition at theta_b = 0.20, eps_T = 0.30 reads
#     8 eta Gm (1-theta_b)(1-(1-theta_b) eps_T)^2 / theta_b = 11.09,
# which gmax = 1.25 (kappa*gmax = 6.25) cannot cover -- hence gmax raised
# to 2.25 (kappa*gmax = 11.25 >= 11.09; margin 1.4%, thinner than the old
# 6.25 vs 5.78). Doob floor drops V0/log(1/theta_b) = 0.3773/1.6094 = 0.234
# (was 0.359 at theta_b = 0.35). Paper Table I values (0.35, 1.25) are now
# STALE until the user's next paper pass; prior certified bound 0.360 is
# superseded by the ladder at this config (see run_e1_acc printout / memo).
# 2026-08-08 design history (user decisions, same day):
#   step 1: theta_b 0.35 -> 0.20 (match Bell), gmax -> 2.25
#   step 2: eps_T 0.30 -> 0.20 (stronger deadline), gmax -> 2.75
#   step 3: eps_T -> 0.0001 after the eps_T=0.01 experiment showed NO
#           closed-loop noise floor (QND: sigma_xi ~ 4 sqrt(eta Gm) xi dies
#           LINEARLY at the target, noise-to-drift ratio constant in xi, so
#           xi contracts at every scale; survivors sit at xi ~ 1e-5 by t=3
#           and exits stay front-loaded). gmax -> 3.90: the edge condition
#           saturates as eps_T -> 0 at LHS -> 19.2, requirement gmax >=
#           3.8394; 3.90 gives kappa*gmax = 19.5, margin 1.6%.
# NOTE the tol_frac=0.05 saturation: eps(T) = eps_T + (eps0-eps_T)*0.05
# -> 0.0351 as eps_T -> 0, deadline level 0.8*eps(T) -> 0.0281. Below
# eps_T ~ 0.01 the terminal claim is set by tol_frac, not eps_T; going
# lower than 0.0001 buys nothing without a longer horizon or smaller
# tol_frac. The binding constraint at this depth is the SPEED LIMIT
# (margin 0.036 at eps_T=0.01, thinner here -- checked before adoption).
# The Doob floor (0.2344) is eps_T-independent throughout.
QUBIT_FUNNEL  = ExpFunnel(eps0=0.70, eps_T=0.0001, T=4.0)
QUBIT_THETA_B = 0.20        # redesigned 2026-08-08 (was 0.35)
QUBIT_LAM     = 0.5         # restored (lam nets out of the corrected bound)
QUBIT_RHO0    = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)

# Shared QP weights (identical across the qubit studies).  gmax=3.90 is the
# edge-defensibility requirement at (theta_b=0.20, eps_T->0); see above.
QUBIT_WEIGHTS = dict(c=20.0, wgamma=1.0, wdelta=1e4, gmax=3.90,
                     regularized=True)

# Hard guard 0: edge defensibility for the qubit design.
_lhs_q = (8.0 * 0.6 * (1.0 - QUBIT_THETA_B)
          * (1.0 - (1.0 - QUBIT_THETA_B) * QUBIT_FUNNEL.eps_T) ** 2
          / QUBIT_THETA_B)
assert _lhs_q <= 5.0 * QUBIT_WEIGHTS["gmax"], (
    f"qubit shell edge not defensible: {_lhs_q:.2f} > kappa*gmax = "
    f"{5.0 * QUBIT_WEIGHTS['gmax']:.2f}"
)

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


# --- Bell study geometry (certified MC, 2026-08-08) -------------------------
#
# The E2 chattering study starts at rho0 = |00><00| (xi0 = 1/2) inside the
# funnel (eps0 = 0.62) but OUTSIDE the design shell:
# (1-theta_b) eps0 = 0.70*0.62 = 0.434 < 0.5, so Corollary III.2 is silent
# for that trajectory -- independent of any envelope -- and the Doob floor
# alone is V0/log(1/theta_b) = 1.6422/1.2040 = 1.36 > 1 (numbers memo,
# 2026-08-06 SS4). The certified study therefore WIDENS the funnel and
# lowers theta_b, keeping the physical |00><00| start:
#
#     eps0 = 0.85, theta_b = 0.20
#       -> shell edge (1-theta_b) eps0 = 0.68 > 0.5      (start inside shell)
#       -> V0 = -log(1 - 0.5/0.85) = 0.8873
#       -> Doob floor V0/log(1/0.2) = 0.8873/1.6094 = 0.551
#
# and the exploratory envelope study (memo SS4) found the slack objective
# <= 0 at every sampled shell state (the three kappa=50 dissipators cover
# the constraint), so the certified bound should land at ~0.55 + Delta with
# Delta tiny -- the Bell analogue of the qubit's 0.360.
#
# Edge defensibility at the new buffer (worst at eps = eps_T):
#     8 eta Gm (1-theta_b)(1-(1-theta_b) eps_T)^2 / theta_b
#       = 8*0.6*0.80*(1-0.80*0.08)^2/0.20 = 16.8 <= kappa*gmax = 50.  OK
#
# H0 = 0 here (paper Table I: rotating frame). systems.bell() defaults to
# omega1 = omega2 = 5, which makes H0 != 0 and mu_xi != 0, contradicting
# Lemma "Bell drift and diffusion" (mu_xi == 0). The certified study zeroes
# both. kappa = 50 per channel matches the E2 figure config (kappa = 5
# cannot stabilize Phi+ at all -- figures.py docstring).

# 2026-08-08 design history (user decisions, same day):
#   step 1: eps0 0.62 -> 0.85, theta_b 0.30 -> 0.20 (certificate speaks),
#           imperfect-preparation start (see bell_rho0_prep)
#   step 2: eps_T 0.08 -> 0.0001 (mirrors the qubit ladder; free at kappa=50)
#   step 3: kappa 50 -> 15 -- the ACTUAL demonstrated dissipative
#           Bell-stabilization rate (Brown et al. 2022: 339 ns ~ 3 MHz =
#           15 Gamma_m at the 200 kHz anchor; the old 50 = 10 MHz was 3.3x
#           demonstrated). Consequences: edge condition (saturates at
#           LHS -> 19.20 as eps_T -> 0) now needs gmax = 1.30 (peak
#           kappa*gmax = 19.5 Gamma_m = 3.9 MHz = 1.3x demonstrated,
#           margin 1.6%), and the T = 2 funnel became speed-limit
#           infeasible in practice (margin 0.004): with the budget only
#           1.6% above the edge requirement, authority and noise pressure
#           nearly cancel at the shell edge, so contraction must slow.
#           Fix chosen (option B): HORIZON T 2 -> 4 (20 us, same as the
#           qubit), halving the contraction demand; speed-limit margin
#           0.036 -- the same deliberate-boundary character as the qubit's
#           0.022. Deadline level stays 0.8*eps(T) = 0.0341 (tol_frac
#           saturation: eps(T) -> 0.05*eps0 = 0.0426 as eps_T -> 0).
BELL_FUNNEL  = ExpFunnel(eps0=0.85, eps_T=0.0001, T=4.0)
BELL_THETA_B = 0.20
BELL_LAM     = 0.5
BELL_DT      = 2.5e-4
BELL_KAPPA   = 15.0

# QP weights: wdelta = 1e3 (Table I Bell column), umax = 1; gmax = 1.30 is
# the edge-defensibility requirement at kappa = 15 (see block comment).
BELL_WEIGHTS = dict(c=20.0, wgamma=1.0, wdelta=1e3, umax=1.0, gmax=1.30,
                    regularized=True)


def bell_system():
    """The certified-study Bell system: rotating frame (H0 = 0), kappa = 50."""
    return bell(omega1=0.0, omega2=0.0,
                kappa=(BELL_KAPPA, BELL_KAPPA, BELL_KAPPA))


def bell_rho0() -> np.ndarray:
    """rho0 = |00><00| (the exact separable start).

    NOTE: from exactly |00><00| the certified closed loop is DETERMINISTIC
    -- |00> lies in the even-parity sector, on which the parity measurement
    is degenerate (H[L]rho = 0, so the innovation multiplies zero), and the
    Markovian law keeps u = 0 exactly (betaH = 0 there, and the coherent
    drives are the only parity-flipping generators). All noise seeds give
    the same path, so a Monte Carlo from this state has no statistical
    content. Kept for reference / the determinism check in the memo; the
    certified MC starts from bell_rho0_prep() below.
    """
    rho0 = np.zeros((4, 4), dtype=complex)
    rho0[0, 0] = 1.0
    return rho0


# Preparation-error tilt for the certified MC: a small x-rotation on the
# first qubit, |psi> = (Rx(theta) x I)|00>. This is the generic experimental
# imperfection -- no real preparation is exact -- and it puts an odd-parity
# component (population sin^2(theta/2) in the Psi sector) into rho0, which
# the parity measurement CAN see: sigma_xi(rho0) != 0, the innovation
# enters, and the ensemble genuinely spreads. xi0 = 1 - cos^2(theta/2)/2.
BELL_THETA_PREP = 0.2          # rad; preparation infidelity sin^2(0.1) ~ 1%


def bell_rho0_prep(theta: float = BELL_THETA_PREP) -> np.ndarray:
    """rho0 = |psi><psi| with |psi> = (Rx(theta) x I)|00>."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    psi = np.array([c, 0.0, -1j * s, 0.0], dtype=complex)   # c|00> - i s|10>
    return np.outer(psi, psi.conj())


# Hard guard 3: buffer fraction proper.
assert 0.0 < BELL_THETA_B < 1.0, (
    f"theta_b={BELL_THETA_B} must lie in (0,1)"
)

# Hard guard 4: Problem 1 precondition for the Bell start, computed with the
# generic infidelity xi = 1 - Tr(Pi rho) (NOT the qubit shortcut 1 - rho00:
# for the Bell target Pi = |Phi+><Phi+|, <00|Pi|00> = 1/2). Checked for the
# PREPARED state actually used by the certified MC.
_sys_bell = bell_system()
_xi0_bell = 1.0 - float(np.trace(_sys_bell.Pi @ bell_rho0_prep()).real)
assert _xi0_bell < (1.0 - BELL_THETA_B) * BELL_FUNNEL.eps0, (
    f"Bell Problem 1 precondition violated: xi(rho0)={_xi0_bell:.3f} must be "
    f"below (1-theta_b) eps0 = {(1.0 - BELL_THETA_B) * BELL_FUNNEL.eps0:.3f}"
)

# Hard guard 5: edge defensibility at the Bell buffer (see docstring above).
_lhs = (8.0 * _sys_bell.eta * 1.0 * (1.0 - BELL_THETA_B)
        * (1.0 - (1.0 - BELL_THETA_B) * BELL_FUNNEL.eps_T) ** 2 / BELL_THETA_B)
assert _lhs <= BELL_KAPPA * BELL_WEIGHTS["gmax"], (
    f"Bell shell edge not defensible: {_lhs:.2f} > kappa*gmax = "
    f"{BELL_KAPPA * BELL_WEIGHTS['gmax']:.2f}"
)
