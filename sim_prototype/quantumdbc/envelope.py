"""Closed-loop certificate evaluation (Corollary IV, closed-loop bound).

The deterministic slack envelope

    delta_V_bar(t) = sup over Omega(t) x [-u_max, u_max] of
                     [ (nu*(rho,t,u_prev)/w_delta - lam V(s,t)) / s ]_+

bounds the closed-loop V-drift pathwise: at the QP optimum the safety
constraint gives E[dV|F_t]/dt <= (delta* - lam V)/s exactly, and the dormant
case contributes <= 0 (covered by the positive part). Corollary III.2 then
yields the a-priori shell-exit bound

    P[tau_Omega <= T] <= (V_0 + int_0^T delta_V_bar dt) / log(1/theta_b),
    V_0 = -log(1 - xi(rho_0)/eps(0)).

Two terms this envelope keeps that a cruder one would discard, and why they
matter (measured on the Table I qubit):

  * the -lam V decay credit: dropping it (bounding E[dV] by delta*/s alone)
    costs a factor ~7 -- the exp-decay strengthening charges lam V into
    delta*, and the certificate must hand it back;
  * the pointwise margin s(rho): dividing the worst nu* by the worst-case
    margin theta_b eps instead couples two suprema that never co-occur.

The envelope also exposes the geometry: on the zero-coherent-gain ridge
(x2 = 0) with gamma saturated, delta* is w_delta-INDEPENDENT (Case III/IV
saturation), so the bound is finite only when the shell edge is defensible,

    8 eta Gamma_m (1-theta_b) (1 - (1-theta_b) eps)^2 / theta_b
        <= kappa gamma_max        (worst at eps = eps_T),

which is what fixes gamma_max in Table I. Once that holds, the remaining
interior slack scales ~1/w_delta and w_delta becomes an effective knob again.

The sup is evaluated on a grid; the grid sup UNDER-estimates the true
envelope, so refine until the bound is stable (the Table I value was checked
Richardson-style over three refinement levels). Qubit only: the QP data
depend on (x2, x3) alone -- mu = 0, betaH = -Omega x2, betaD = -kappa xi,
sigma = -sqrt(eta Gamma_m)(1 - x3^2), xi = (1-x3)/2; x1 enters no
coefficient -- so Omega(t) reduces to the half-disc x2^2 + x3^2 <= 1,
x3 >= 1 - 2(1-theta_b) eps(t). nu* is piecewise affine in u_prev, so a
small u_prev grid including the endpoints carries the sup.
"""
from __future__ import annotations
import numpy as np

from .systems import System
from .controller import QPData, solve_closed_form
from .coefficients import _drive_amp, _kappa, _meas_rate_qubit


def qubit_slack_envelope(sys: System, cfg, t_grid=None,
                         n_x3: int = 161, n_x2: int = 81,
                         u_prev_grid=(-1.0, -0.5, 0.0, 0.5, 1.0)):
    """Grid evaluation of delta_V_bar(t) on the qubit design shell.

    Parameters
    ----------
    sys    : the qubit System (operators supply Omega, Gamma_m, kappa)
    cfg    : SimConfig -- supplies theta_b, lam, weights, bounds, funnel
    t_grid : time grid; defaults to 81 points on [0, funnel.T]
    n_x3, n_x2 : shell-grid resolution (x3 radial in xi, x2 coherence)
    u_prev_grid : carried-command grid, scaled by cfg.umax

    Returns
    -------
    (t_grid, delta_V_bar) with delta_V_bar[k] the shell-grid sup at t_grid[k].
    """
    if sys.name != "qubit":
        raise ValueError("the grid envelope is implemented for the qubit "
                         "instance only (15-dim Bell shell is not griddable)")
    fun = cfg.funnel
    if t_grid is None:
        t_grid = np.linspace(0.0, fun.T, 81)
    t_grid = np.asarray(t_grid, dtype=float)

    Om = _drive_amp(sys.Hc[0])
    Gm = _meas_rate_qubit(sys.L)
    ka = _kappa(sys.Lc[0])
    se = np.sqrt(sys.eta * Gm)

    wgamma = np.array([cfg.wgamma])
    umax_v = np.array([cfg.umax])
    gmax_v = np.array([cfg.gmax])
    u_prevs = tuple(cfg.umax * u for u in u_prev_grid)

    dbar = np.zeros_like(t_grid)
    for k, t in enumerate(t_grid):
        eps_t = float(fun.eps(t))
        eps_dot = float(fun.eps_dot(t))
        # shell in x3: xi <= (1-theta_b) eps_t  <=>  x3 >= 1 - 2(1-theta_b) eps_t
        x3_min = max(-1.0, 1.0 - 2.0 * (1.0 - cfg.theta_b) * eps_t)
        best = 0.0
        for x3 in np.linspace(x3_min, 1.0, n_x3):
            xi = 0.5 * (1.0 - x3)
            s = max(eps_t - xi, cfg.theta_b * eps_t)
            V = -np.log(s / eps_t)
            kap_V = 1.0 / s
            sigma = -se * (1.0 - x3 * x3)
            alpha = (-(xi / eps_t) * eps_dot          # mu = 0 for the qubit
                     + 0.5 * kap_V * sigma * sigma)
            betaD = np.array([-ka * xi])
            x2_max = np.sqrt(max(0.0, 1.0 - x3 * x3))
            for x2 in np.linspace(-x2_max, x2_max, n_x2):
                betaH = np.array([-Om * x2])
                if cfg.regularized:
                    wu = np.array([cfg.c / (abs(betaH[0]) + cfg.eps_f)])
                    wr = np.array([cfg.wr])
                else:
                    wu = np.array([1.0])
                    wr = np.array([0.0])
                for up in u_prevs:
                    d = QPData(alpha=alpha, betaH=betaH, betaD=betaD,
                               V=V, lam=cfg.lam, u_prev=np.array([up]),
                               wu=wu, wr=wr, wgamma=wgamma,
                               wdelta=cfg.wdelta,
                               umax=umax_v, gmax=gmax_v)
                    res = solve_closed_form(d)
                    q = (res.delta - cfg.lam * V) / s
                    if q > best:
                        best = q
        dbar[k] = best
    return t_grid, dbar


def closed_loop_bound(cfg, t_grid, delta_V_bar, xi0: float):
    """Assemble the a-priori shell-exit bound from the slack envelope.

    Returns (bound, parts) where parts holds V0, Delta (the integrated
    envelope), and the exit level log(1/theta_b). The bound is an envelope:
    bound > 1 means the certificate is silent at these parameters -- with
    the shell edge defensible (see module docstring) and w_delta ~ 1e4 the
    Table I qubit gives bound ~ 0.53.
    """
    fun = cfg.funnel
    Delta = float(np.trapezoid(np.asarray(delta_V_bar, dtype=float), t_grid))
    V0 = float(-np.log(1.0 - xi0 / fun.eps0))
    level = float(np.log(1.0 / cfg.theta_b))
    return (V0 + Delta) / level, dict(V0=V0, Delta=Delta, level=level)
