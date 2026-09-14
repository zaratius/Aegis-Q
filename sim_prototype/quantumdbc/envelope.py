"""Closed-loop certificate evaluation (Corollary IV, closed-loop bound).

The deterministic slack envelope

    delta_V_bar(t) = sup over Omega(t) of
                     [ (nu*(rho,t)/w_delta - lam V(s,t)) / s ]_+

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
x3 >= 1 - 2(1-theta_b) eps(t).
"""
from __future__ import annotations
import numpy as np

from .systems import System
from .controller import QPData, solve_closed_form, solve_multichannel
from .coefficients import _drive_amp, _kappa, _meas_rate_qubit

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                                          # pragma: no cover
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @njit(parallel=True, cache=True)
    def _bell_dbar_kernel(eps_arr, epsdot_arr, frac_xi, n_p1, n_p2,
                          kap0, kap1, kap2, se2, theta_b, lam,
                          wgamma, wdelta, gmax):
        """Scalar re-implementation of the Bell envelope inner loops.

        Arithmetic mirrors the NumPy reference path (same screen, same
        shared bisection bracket per (t, xi) group, same 100 halvings);
        parity is pinned in test_bell_envelope.py. prange over the time
        grid -- each t is independent. Pure scalar math, no BLAS inside,
        so the BLAS thread pins do not apply here.
        """
        nt = eps_arr.shape[0]
        dbar = np.zeros(nt)
        smax = np.full(nt, -np.inf)
        nsol = np.zeros(nt, np.int64)
        for k in prange(nt):
            eps_t = eps_arr[k]
            eps_dot = epsdot_arr[k]
            xi_max = (1.0 - theta_b) * eps_t
            best = 0.0
            sm = -np.inf
            ns = 0
            for m in range(frac_xi.shape[0]):
                xi = frac_xi[m] * xi_max
                s = eps_t - xi
                s_min = theta_b * eps_t
                if s < s_min:
                    s = s_min
                V = -np.log(s / eps_t)
                nu_c = wdelta * lam * V
                # pass 1: screen g(nu_c); record the survivors' max alpha
                # (shared bisection bracket, as in the NumPy path)
                any_pos = False
                amax = -np.inf
                for i in range(n_p1):
                    f1 = i / (n_p1 - 1.0)
                    P1 = f1 * xi
                    Pr = xi - P1
                    sigma2 = se2 * (1.0 - xi) * (1.0 - xi) * Pr * Pr
                    alpha = -(xi / eps_t) * eps_dot + 0.5 * sigma2 / s
                    for j in range(n_p2):
                        f2 = j / (n_p2 - 1.0)
                        P2 = f2 * Pr
                        P3 = Pr - P2
                        g = alpha
                        gam = nu_c * kap0 * P1 / wgamma
                        if gam > gmax:
                            gam = gmax
                        g -= kap0 * P1 * gam
                        gam = nu_c * kap1 * P2 / wgamma
                        if gam > gmax:
                            gam = gmax
                        g -= kap1 * P2 * gam
                        gam = nu_c * kap2 * P3 / wgamma
                        if gam > gmax:
                            gam = gmax
                        g -= kap2 * P3 * gam
                        if g > sm:
                            sm = g
                        if g > 0.0:
                            any_pos = True
                            if alpha > amax:
                                amax = alpha
                if not any_pos:
                    continue
                hi0 = wdelta * (amax + lam * V) + 1.0
                # pass 2: bisect the survivors on the monotone residual
                for i in range(n_p1):
                    f1 = i / (n_p1 - 1.0)
                    P1 = f1 * xi
                    Pr = xi - P1
                    sigma2 = se2 * (1.0 - xi) * (1.0 - xi) * Pr * Pr
                    alpha = -(xi / eps_t) * eps_dot + 0.5 * sigma2 / s
                    for j in range(n_p2):
                        f2 = j / (n_p2 - 1.0)
                        P2 = f2 * Pr
                        P3 = Pr - P2
                        g = alpha
                        gam = nu_c * kap0 * P1 / wgamma
                        if gam > gmax:
                            gam = gmax
                        g -= kap0 * P1 * gam
                        gam = nu_c * kap1 * P2 / wgamma
                        if gam > gmax:
                            gam = gmax
                        g -= kap1 * P2 * gam
                        gam = nu_c * kap2 * P3 / wgamma
                        if gam > gmax:
                            gam = gmax
                        g -= kap2 * P3 * gam
                        if g <= 0.0:
                            continue
                        ns += 1
                        lo = nu_c
                        hi = hi0
                        for _ in range(100):
                            mid = 0.5 * (lo + hi)
                            r = alpha + lam * V - mid / wdelta
                            gam = mid * kap0 * P1 / wgamma
                            if gam > gmax:
                                gam = gmax
                            r -= kap0 * P1 * gam
                            gam = mid * kap1 * P2 / wgamma
                            if gam > gmax:
                                gam = gmax
                            r -= kap1 * P2 * gam
                            gam = mid * kap2 * P3 / wgamma
                            if gam > gmax:
                                gam = gmax
                            r -= kap2 * P3 * gam
                            if r > 0.0:
                                lo = mid
                            else:
                                hi = mid
                        nu = 0.5 * (lo + hi)
                        q = (nu / wdelta - lam * V) / s
                        if q > best:
                            best = q
            dbar[k] = best
            smax[k] = sm
            nsol[k] = ns
        return dbar, smax, nsol


def qubit_slack_envelope(sys: System, cfg, t_grid=None,
                         n_x3: int = 161, n_x2: int = 81):
    """Grid evaluation of delta_V_bar(t) on the qubit design shell.

    Parameters
    ----------
    sys    : the qubit System (operators supply Omega, Gamma_m, kappa)
    cfg    : SimConfig -- supplies theta_b, lam, weights, bounds, funnel
    t_grid : time grid; defaults to 81 points on [0, funnel.T]
    n_x3, n_x2 : shell-grid resolution (x3 radial in xi, x2 coherence)

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

    dbar = np.zeros_like(t_grid)
    for k, t in enumerate(t_grid):
        eps_t = float(fun.eps(t))
        eps_dot = float(fun.eps_dot(t))
        # shell in x3: xi <= (1-theta_b) eps_t  <=>  x3 >= 1 - 2(1-theta_b) eps_t
        x3_min = max(-1.0, 1.0 - 2.0 * (1.0 - cfg.theta_b) * eps_t)
        # linear grid PLUS a log-spaced layer near x3 = 1 (xi -> 0): the
        # positive sliver of the objective sits at xi ~ sqrt(wgamma/wdelta)
        # /kappa, where the quadratic dissipative stiffness falls below the
        # 1/wdelta term -- same boundary layer as the Bell envelope.
        xi_max_t = (1.0 - cfg.theta_b) * eps_t
        x3_lin = np.linspace(x3_min, 1.0, n_x3)
        x3_log = 1.0 - 2.0 * xi_max_t * np.geomspace(1e-6, 0.05, n_x3 // 2)
        x3_grid = np.unique(np.concatenate([x3_lin, x3_log]))
        best = 0.0
        for x3 in x3_grid:
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
                else:
                    wu = np.array([1.0])
                d = QPData(alpha=alpha, betaH=betaH, betaD=betaD,
                           V=V, lam=cfg.lam,
                           wu=wu, wgamma=wgamma,
                           wdelta=cfg.wdelta,
                           umax=umax_v, gmax=gmax_v)
                res = solve_closed_form(d)
                q = (res.delta - cfg.lam * V) / s
                if q > best:
                    best = q
        dbar[k] = best
    return t_grid, dbar


def bell_slack_envelope(sys: System, cfg, t_grid=None,
                        n_xi: int = 161, n_p1: int = 81, n_p2: int = 41,
                        backend: str = "numba"):
    """Grid evaluation of delta_V_bar(t) on the Bell design shell.

    The 15-dim Bell shell is not griddable directly, but it does not need
    to be: with H0 = 0 (rotating frame) the QP data at (rho, t) depend on
    rho only through four scalars that live on a simplex,

        xi   = 1 - <Phi+|rho|Phi+>,
        P_k  = <psi_k|rho|psi_k>,  psi_k in {Phi-, Psi+, Psi-},
        with P_1 + P_2 + P_3 = xi exactly (the Bell basis is an ONB),

    because each coefficient is an exact function of them:

        mu       == 0                                  (H0 = 0, QND parity L)
        betaD_k  = -kappa_k P_k                        (Lemma "Bell diss. gains")
        sigma    = -4 sqrt(eta Gm) (1 - xi)(P_2 + P_3)
                 (L = sqrt(Gm) U with U = sz x sz Hermitian unitary and
                  Pi U = U Pi = Pi:  Tr(Pi H[L]rho) = 2 sqrt(Gm)(1-xi)(1-<U>),
                  and 1 - <U> = 2(P_2 + P_3) since the Psi sector carries
                  parity -1)

    and the coherent gains betaH_i only ever help: in the dual residual
    g(nu) each coherent term is -min(nu betaH^2/wu, |betaH| umax), pointwise
    non-increasing in |betaH| (both branches increase in |betaH|, including
    through the state weight wu = c/(|betaH|+eps_f)), so the root nu* -- and
    with it delta* and the objective -- is maximized at betaH = 0. betaH = 0
    is attained on the simplex slice itself (Bell-diagonal states commute
    with Pi), so restricting to that slice loses nothing: the slice sup IS
    the shell sup. The grid still under-estimates within the slice; refine
    until the bound is stable, as for the qubit.

    A vectorized pre-screen avoids the multichannel root find almost
    everywhere: nu* > wdelta lam V (needed for a positive objective) iff
    g(wdelta lam V) > 0, and g at that single nu is closed-form. Points
    failing the screen contribute 0; the root find runs only on survivors.

    Returns (t_grid, delta_V_bar, diag) with diag holding the screen margin
    max_t max_shell g(nu_c) (<= 0 everywhere means delta_V_bar == 0 exactly
    on the grid) and the count of root-find calls.

    backend: "numba" (default; prange over the t grid, falls back to the
    NumPy path when numba is unavailable) or "python" (the vectorized
    NumPy reference; parity between the two is pinned in the test suite).
    """
    if sys.name != "bell":
        raise ValueError("bell_slack_envelope requires the Bell instance")
    if float(np.max(np.abs(sys.H0))) != 0.0:
        raise ValueError("bell_slack_envelope assumes H0 = 0 (rotating "
                         "frame): mu_xi == 0 enters the feature reduction")
    fun = cfg.funnel
    if t_grid is None:
        t_grid = np.linspace(0.0, fun.T, 81)
    t_grid = np.asarray(t_grid, dtype=float)

    # kappa_k = Tr(Lc^dag Lc): exact for Lc = sqrt(kappa)|Phi+><psi_k| (unit
    # vectors), where the qubit helper _kappa = max|Lc|^2 would return
    # kappa/4 (the entries are sqrt(kappa)/2) and understate the dissipators.
    kap = np.array([float(np.trace(Lc.conj().T @ Lc).real) for Lc in sys.Lc])
    Gm = float(np.abs(sys.L[0, 0]) ** 2)                   # L = sqrt(Gm) sz x sz
    se2 = 16.0 * sys.eta * Gm                              # sigma^2 prefactor

    if backend == "numba" and _HAVE_NUMBA:
        # xi grid as fractions of the per-t shell edge: linear + log layer
        # (same construction as the NumPy path, up to ULP placement)
        frac_xi = np.unique(np.concatenate([
            np.linspace(0.0, 1.0, n_xi),
            np.geomspace(1e-6, 0.05, n_xi // 2)]))
        eps_arr = np.array([float(fun.eps(t)) for t in t_grid])
        epsdot_arr = np.array([float(fun.eps_dot(t)) for t in t_grid])
        dbar, smax, nsol = _bell_dbar_kernel(
            eps_arr, epsdot_arr, frac_xi, n_p1, n_p2,
            float(kap[0]), float(kap[1]), float(kap[2]), se2,
            cfg.theta_b, cfg.lam, cfg.wgamma, cfg.wdelta, cfg.gmax)
        return t_grid, dbar, dict(screen_max=float(np.max(smax)),
                                  n_solves=int(np.sum(nsol)))

    # simplex fractions (broadcast grid): P1 = f1*xi, P2 = f2*(xi - P1)
    f1 = np.linspace(0.0, 1.0, n_p1)[:, None]              # (n_p1, 1)
    f2 = np.linspace(0.0, 1.0, n_p2)[None, :]              # (1, n_p2)

    dbar = np.zeros_like(t_grid)
    screen_max = -np.inf
    n_solves = 0
    for k, t in enumerate(t_grid):
        eps_t = float(fun.eps(t))
        eps_dot = float(fun.eps_dot(t))
        xi_max = (1.0 - cfg.theta_b) * eps_t
        # xi grid: linear across the shell PLUS a log-spaced boundary layer
        # near xi = 0. The positive sliver of the objective lives at
        # xi ~ sqrt(wgamma/wdelta)/kappa (where the quadratic dissipative
        # stiffness sum (kappa_k P_k)^2/wgamma falls below the 1/wdelta
        # term); a linear grid at any affordable resolution steps over it.
        xi_lin = np.linspace(0.0, xi_max, n_xi)
        xi_log = xi_max * np.geomspace(1e-6, 0.05, n_xi // 2)
        xi_grid = np.unique(np.concatenate([xi_lin, xi_log]))
        best = 0.0
        for xi in xi_grid:
            s = max(eps_t - xi, cfg.theta_b * eps_t)
            V = -np.log(s / eps_t)
            nu_c = cfg.wdelta * cfg.lam * V
            P1 = f1 * xi                                   # (n_p1, 1)
            Prest = xi - P1                                # (n_p1, 1)
            P2 = f2 * Prest                                # (n_p1, n_p2)
            P3 = Prest - P2                                # (n_p1, n_p2)
            sigma2 = se2 * (1.0 - xi) ** 2 * Prest ** 2    # exact, (n_p1, 1)
            alpha = -(xi / eps_t) * eps_dot + 0.5 * sigma2 / s
            # screen: g(nu_c) = alpha + sum_k betaD_k gamma_k(nu_c)
            #   with gamma_k = clip(nu_c kap_k P_k / wgamma, 0, gmax)
            g_c = alpha.copy() * np.ones_like(P2)
            for Pk, ka_k in ((P1 * np.ones_like(P2), kap[0]),
                             (P2, kap[1]), (P3, kap[2])):
                gam = np.minimum(nu_c * ka_k * Pk / cfg.wgamma, cfg.gmax)
                g_c -= ka_k * Pk * gam
            m = float(np.max(g_c))
            if m > screen_max:
                screen_max = m
            if m <= 0.0:
                continue                                   # objective <= 0 here
            # survivors: vectorized bisection on the monotone dual residual
            # (betaH = 0, so the residual is closed-form array math; agreement
            # with solve_multichannel is pinned in test_bell_envelope.py)
            mask = g_c > 0.0
            a_f = np.broadcast_to(alpha, g_c.shape)[mask]
            P_f = [np.broadcast_to(P1, g_c.shape)[mask],
                   np.broadcast_to(P2, g_c.shape)[mask],
                   np.broadcast_to(P3, g_c.shape)[mask]]
            n_solves += int(a_f.size)

            def g_vec(nu):
                out = a_f + cfg.lam * V - nu / cfg.wdelta
                for Pk, ka_k in zip(P_f, kap):
                    gam = np.minimum(nu * ka_k * Pk / cfg.wgamma, cfg.gmax)
                    out = out - ka_k * Pk * gam
                return out

            # bracket: g(nu) <= Theta - nu/wdelta < 0 for nu > wdelta*Theta
            hi0 = cfg.wdelta * (float(np.max(a_f)) + cfg.lam * V) + 1.0
            lo = np.full(a_f.shape, nu_c)
            hi = np.full(a_f.shape, hi0)
            for _ in range(100):                           # 2^-100 * hi0: exact
                mid = 0.5 * (lo + hi)
                pos = g_vec(mid) > 0.0
                lo = np.where(pos, mid, lo)
                hi = np.where(pos, hi, mid)
            nu_star = 0.5 * (lo + hi)
            q = (nu_star / cfg.wdelta - cfg.lam * V) / s
            qm = float(np.max(q))
            if qm > best:
                best = qm
        dbar[k] = best
    return t_grid, dbar, dict(screen_max=float(screen_max),
                              n_solves=int(n_solves))


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
