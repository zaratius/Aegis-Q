"""
Minimum-effort safe controller (Section IV).

Three solvers, all for the Markovian barrier-QP: at each (rho, t),

    min (wu/2) u^2 + (wgamma/2) gamma^2 + (wdelta/2) delta^2
    s.t. alpha + betaH u + betaD gamma <= -lam V + delta,  boxes,

with the state-dependent weight wu = c/(|betaH| + eps_f).  The law
depends on (rho, t) only; there is no carried previous command.

* ``solve_closed_form``  -- the literal five-case decision tree of
  Theorem IV.2, for the single-channel case (m = p = 1).  This is the
  controller used for the qubit and the object validated against OSQP.

* ``solve_multichannel`` -- the master equation (38) solved as a monotone
  one-dimensional root find in the safety multiplier nu; valid for any
  m, p, hence used for the qutrit and Bell instances.  For m = p = 1 it
  reproduces ``solve_closed_form`` (checked in tests).

* ``solve_osqp``         -- reference solution of the same QP via OSQP,
  for cross-validation only (never called in the closed loop).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class QPData:
    """Pointwise QP data at one (rho, t)."""
    alpha: float                 # uncontrolled drift  (22)
    betaH: np.ndarray            # coherent gains      (len m)
    betaD: np.ndarray            # dissipative gains   (len p), each < 0
    V: float                     # barrier value
    lam: float                   # exponential decay rate
    wu: np.ndarray               # state-dependent coherent weights (len m)
    wgamma: np.ndarray           # dissipative weights (len p)
    wdelta: float                # slack penalty
    umax: np.ndarray             # coherent bounds (len m)
    gmax: np.ndarray             # dissipative bounds (len p)


@dataclass
class QPResult:
    u: np.ndarray
    gamma: np.ndarray
    delta: float
    nu: float
    case: str


# --------------------------------------------------------------------------
# Theorem IV.2 -- five-case closed form (single channel)
# --------------------------------------------------------------------------
def solve_closed_form(d: QPData) -> QPResult:
    """Exact five-case solution of the QP for m = p = 1."""
    if d.betaH.size != 1 or d.betaD.size != 1:
        raise ValueError("solve_closed_form requires m = p = 1")
    bH = float(d.betaH[0]); bD = float(d.betaD[0])
    wu = float(d.wu[0])
    wg = float(d.wgamma[0]); wd = d.wdelta
    umax = float(d.umax[0]); gmax = float(d.gmax[0])
    lamV = d.lam * d.V

    # ---- Lemma IV.1: dormancy test --------------------------------------
    Theta = d.alpha + lamV
    if Theta <= 0.0:
        return QPResult(np.array([0.0]), np.array([0.0]), 0.0, 0.0, "V")

    # ---- Case I candidate ----------------------------------------------
    N_I = Theta
    D_I = bH ** 2 / wu + bD ** 2 / wg + 1.0 / wd
    nu_I = N_I / D_I
    u_I = -nu_I * bH / wu
    g_I = -nu_I * bD / wg

    u_ok = abs(u_I) <= umax
    g_ok = g_I <= gmax

    if u_ok and g_ok:
        return QPResult(np.array([u_I]), np.array([g_I]),
                        nu_I / wd, nu_I, "I")

    # ---- Case II: u saturated ------------------------------------------
    if (not u_ok) and g_ok:
        usat = umax * np.sign(u_I)
        N_II = d.alpha + bH * usat + lamV
        D_II = bD ** 2 / wg + 1.0 / wd
        nu_II = N_II / D_II
        g_II = -nu_II * bD / wg
        if g_II <= gmax:
            return QPResult(np.array([usat]), np.array([g_II]),
                            nu_II / wd, nu_II, "II")
        return _case_IV(d, bH, bD, wd, umax, gmax, lamV, usat)

    # ---- Case III: gamma saturated -------------------------------------
    if u_ok and (not g_ok):
        N_III = d.alpha + bD * gmax + lamV
        D_III = bH ** 2 / wu + 1.0 / wd
        nu_III = N_III / D_III
        u_III = -nu_III * bH / wu
        if abs(u_III) <= umax:
            return QPResult(np.array([u_III]), np.array([gmax]),
                            nu_III / wd, nu_III, "III")
        usat = umax * np.sign(u_III)
        return _case_IV(d, bH, bD, wd, umax, gmax, lamV, usat)

    # ---- Case IV: both saturated ---------------------------------------
    usat = umax * np.sign(u_I)
    return _case_IV(d, bH, bD, wd, umax, gmax, lamV, usat)


def _case_IV(d, bH, bD, wd, umax, gmax, lamV, usat) -> QPResult:
    N_IV = d.alpha + bH * usat + bD * gmax + lamV
    nu_IV = N_IV * wd
    return QPResult(np.array([usat]), np.array([gmax]),
                    nu_IV / wd, nu_IV, "IV")


# --------------------------------------------------------------------------
# Master equation (38) as a monotone 1-D root find -- multi-channel
# --------------------------------------------------------------------------
def _primal_from_nu(d: QPData, nu: float):
    """Box-projected stationary primal for a given multiplier nu >= 0."""
    u = -nu * d.betaH / d.wu
    u = np.clip(u, -d.umax, d.umax)
    gamma = -nu * d.betaD / d.wgamma            # betaD < 0  =>  >= 0
    gamma = np.clip(gamma, 0.0, d.gmax)
    delta = nu / d.wdelta
    return u, gamma, delta


def _residual(d: QPData, nu: float) -> float:
    """g(nu) = alpha + sum betaH u + sum betaD gamma + lam V - delta.
    The safety constraint is g(nu) <= 0; g is non-increasing in nu."""
    u, gamma, delta = _primal_from_nu(d, nu)
    return (d.alpha + float(d.betaH @ u) + float(d.betaD @ gamma)
            + d.lam * d.V - delta)


def solve_multichannel(d: QPData, tol: float = 1e-12) -> QPResult:
    """Solve the QP (46) for arbitrary m, p via the dual root find."""
    g0 = _residual(d, 0.0)
    if g0 <= 0.0:                                    # dormant: nu* = 0
        u, gamma, delta = _primal_from_nu(d, 0.0)
        return QPResult(u, gamma, 0.0, 0.0, "V")

    # bracket: g decreases to -inf as nu grows (delta term dominates)
    nu_hi = 1.0
    for _ in range(200):
        if _residual(d, nu_hi) < 0.0:
            break
        nu_hi *= 2.0
    else:
        raise RuntimeError("failed to bracket the safety multiplier")

    lo, hi = 0.0, nu_hi
    for _ in range(200):                             # bisection on monotone g
        mid = 0.5 * (lo + hi)
        if _residual(d, mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    nu = 0.5 * (lo + hi)
    u, gamma, delta = _primal_from_nu(d, nu)

    n_sat = int(np.sum(np.abs(np.abs(u) - d.umax) < 1e-9)
                + np.sum(np.abs(gamma - d.gmax) < 1e-9))
    case = {0: "I", }.get(n_sat, f"sat({n_sat})")
    return QPResult(u, gamma, delta, nu, case)


# --------------------------------------------------------------------------
# OSQP reference solve of the QP (46) -- cross-validation only
# --------------------------------------------------------------------------
def solve_osqp(d: QPData) -> QPResult:
    """Solve the same QP with OSQP.  Used only to validate the closed form."""
    import osqp
    from scipy import sparse

    m, p = d.betaH.size, d.betaD.size
    n = m + p + 1                                    # z = (u, gamma, delta)

    P = sparse.diags(np.concatenate([d.wu, d.wgamma, [d.wdelta]]),
                     format="csc")
    q = np.zeros(n)

    rows = []
    lo = []
    hi = []
    # safety constraint: sum betaH u + sum betaD gamma - delta <= -lamV - alpha
    safety = np.concatenate([d.betaH, d.betaD, [-1.0]])
    rows.append(safety); lo.append(-np.inf)
    hi.append(-d.lam * d.V - d.alpha)
    # box constraints
    for i in range(m):
        e = np.zeros(n); e[i] = 1.0
        rows.append(e); lo.append(-d.umax[i]); hi.append(d.umax[i])
    for j in range(p):
        e = np.zeros(n); e[m + j] = 1.0
        rows.append(e); lo.append(0.0); hi.append(d.gmax[j])
    e = np.zeros(n); e[-1] = 1.0
    rows.append(e); lo.append(0.0); hi.append(np.inf)

    A = sparse.csc_matrix(np.array(rows))
    prob = osqp.OSQP()
    prob.setup(P, q, A, np.array(lo), np.array(hi),
               eps_abs=1e-8, eps_rel=1e-8, max_iter=600000,
               polishing=True, polish_refine_iter=15,
               scaling=50, adaptive_rho=False, rho=1.0, verbose=False)
    res = prob.solve()
    ok = res.info.status_val in (1, 2)
    if not ok:                  # first-order method on an ill-conditioned QP:
        ok = (res.info.prim_res < 2e-5 and res.info.dual_res < 2e-5)
    if not ok:
        raise RuntimeError(f"OSQP status: {res.info.status} "
                           f"(prim {res.info.prim_res:.1e}, "
                           f"dual {res.info.dual_res:.1e})")
    z = res.x
    return QPResult(z[:m], z[m:m + p], float(z[-1]),
                    float(res.y[0]), "osqp")
