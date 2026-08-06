"""
Closed-loop driver: Algorithm 1 of the manuscript.

Each step performs (i) coefficient evaluation, (ii) closed-form QP solution,
(iii) Milstein propagation.  The law is Markovian: each step's QP depends
on the current (rho, t) only.

The controller may be run on a *model* that differs from the true plant
(the ``design_sys`` argument of ``run_trajectory``).  This realises the
robust / parametrically-mismatched controller of Proposition IV.3 and the
Section V-E8 study: the QP coefficients are evaluated on the design model
while the state is propagated by the true plant.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .systems import System
from .barrier import FunnelGaugeBarrier, ExpFunnel
from .coefficients import coefficients_generic, infidelity, to_bloch, from_bloch
from .controller import QPData, solve_closed_form, solve_multichannel
from .integrator import milstein_step, project_physical
from . import kernel as _kernel


@dataclass
class SimConfig:
    """Closed-loop configuration."""
    funnel: ExpFunnel
    lam: float = 1.0                 # exponential barrier-decay rate
    theta_b: float = 0.10            # relative buffer: s_b(t) = theta_b * eps(t)
    c: float = 1.0                   # state-weight scale
    eps_f: float = 1e-2              # state-weight floor
    wgamma: float = 1.0              # dissipative weight
    wdelta: float = 1e3              # slack penalty
    umax: float = 1.0                # coherent bound (per channel)
    gmax: float = 1.0                # dissipative bound (per channel)
    dt: float = 1e-3
    regularized: bool = True         # False -> w_u == 1 (canonical CBF-QP baseline)
    project: bool = True             # positivity-preserving projection each step


@dataclass
class Trajectory:
    t: np.ndarray
    xi: np.ndarray
    eps: np.ndarray
    eps_sb: np.ndarray
    u: np.ndarray                    # (steps, m)
    gamma: np.ndarray                # (steps, p)
    delta: np.ndarray
    nu: np.ndarray
    betaH: np.ndarray                # (steps, m)  -- as seen by the controller
    betaD: np.ndarray                # (steps, p)  -- as seen by the controller
    alpha: np.ndarray                # (steps,)  uncontrolled QP drift
    case: list = field(default_factory=list)
    max_neg: float = 0.0             # worst eigenvalue negativity of rho
    confined: bool = True            # xi < eps for all t


def _weights(cfg: SimConfig, betaH: np.ndarray):
    """State-dependent coherent weight, eq. (31); or unregularised (w_u=1)."""
    m = betaH.size
    if cfg.regularized:
        wu = cfg.c / (np.abs(betaH) + cfg.eps_f)
    else:
        wu = np.ones(m)
    return wu


_CASE_SINGLE = {-1: "V", 1: "I", 2: "II", 3: "III", 4: "IV"}


def _extract_system_arrays(sys: System) -> dict:
    """Stack a System's operator data into contiguous complex128 arrays and
    precompute the daggers / J^dag J products the kernel consumes."""
    cc = np.ascontiguousarray
    N = sys.N
    Lc = cc(np.stack(sys.Lc).astype(np.complex128))
    return {
        "N": N, "NG": N * N - 1, "m": sys.m, "p": sys.p, "eta": float(sys.eta),
        "G": cc(np.stack(sys.gens).astype(np.complex128)),
        "I_over_N": cc((np.eye(N) / N).astype(np.complex128)),
        "Pi": cc(sys.Pi.astype(np.complex128)),
        "H0": cc(sys.H0.astype(np.complex128)),
        "L": cc(sys.L.astype(np.complex128)),
        "Ld": cc(sys.L.conj().T.astype(np.complex128)),
        "LdL": cc((sys.L.conj().T @ sys.L).astype(np.complex128)),
        "Hc": cc(np.stack(sys.Hc).astype(np.complex128)),
        "Lc": Lc,
        "Lcd": cc(np.stack([lc.conj().T for lc in sys.Lc]).astype(np.complex128)),
        "LcdLc": cc(np.stack([lc.conj().T @ lc for lc in sys.Lc]).astype(np.complex128)),
    }


def run_trajectory(sys: System, cfg: SimConfig, rho0: np.ndarray,
                   seed: int = 0, store: bool = True,
                   design_sys: System | None = None,
                   backend: str = "numba",
                   t_stop: float | None = None) -> Trajectory:
    """Integrate one closed-loop sample path (Algorithm 1).

    Identical interface and result to the reference implementation; by default
    it runs the compiled Numba kernel (``quantumdbc.kernel``), which reproduces
    the reference trajectory to ~machine precision under a given ``seed``.  Pass
    ``backend="python"`` to force the pure-Python reference path (used for
    cross-validation, and the automatic fallback when Numba is unavailable).
    ``t_stop`` caps the integration at ``round(t_stop/dt)`` steps while keeping
    the full funnel geometry (rate set by ``funnel.T``) -- an early stop for
    exit-rate sweeps where all exits occur well before the horizon.
    See ``_run_trajectory_python`` for the full parameter documentation.
    """
    if backend != "numba" or not _kernel.HAVE_NUMBA:
        return _run_trajectory_python(sys, cfg, rho0, seed=seed, store=store,
                                      design_sys=design_sys, t_stop=t_stop)

    ctrl = sys if design_sys is None else design_sys
    if ctrl is not sys and (ctrl.N, ctrl.m, ctrl.p) != (sys.N, sys.m, sys.p):
        raise ValueError("design_sys must match the plant dimension "
                         "and channel counts")

    P = _extract_system_arrays(sys)
    C = P if ctrl is sys else _extract_system_arrays(ctrl)
    f = cfg.funnel
    N, NG, m, p = P["N"], P["NG"], P["m"], P["p"]
    n_steps = int(round((f.T if t_stop is None else t_stop) / cfg.dt))
    single = 1 if (m == 1 and p == 1) else 0

    params = np.array([
        cfg.dt, P["eta"], C["eta"], f.eps0, f.eps_T, f.r, cfg.theta_b,
        cfg.lam, cfg.c, cfg.eps_f, 0.0, cfg.umax, cfg.gmax,
        1e-12, 1e-6, cfg.wgamma, cfg.wdelta], np.float64)
    # slot 10 (0.0) is retired: it carried the temporal weight w_r before the
    # law went Markovian; kept as a dead slot so kernel indices stay stable.
    ints = np.array([n_steps, N, NG, m, p, single,
                     1 if cfg.project else 0, 1 if cfg.regularized else 0],
                    np.int64)

    x0 = to_bloch(sys, rho0).astype(np.float64)
    dW = np.random.default_rng(seed).normal(0.0, np.sqrt(cfg.dt), size=n_steps)

    xi_o = np.empty(n_steps); eps_o = np.empty(n_steps)
    epssb_o = np.empty(n_steps); delta_o = np.empty(n_steps)
    nu_o = np.empty(n_steps); alpha_o = np.empty(n_steps)
    u_o = np.empty((n_steps, m)); g_o = np.empty((n_steps, p))
    bH_o = np.empty((n_steps, m)); bD_o = np.empty((n_steps, p))
    case_o = np.empty(n_steps, np.int64)

    max_neg, confined = _kernel.run_traj_kernel(
        x0, dW, params, ints,
        P["G"], P["I_over_N"], P["Pi"], P["H0"], P["L"], P["Ld"], P["LdL"],
        P["Hc"], P["Lc"], P["Lcd"], P["LcdLc"],
        C["Pi"], C["H0"], C["L"], C["Ld"], C["LdL"],
        C["Hc"], C["Lc"], C["Lcd"], C["LcdLc"],
        xi_o, eps_o, epssb_o, u_o, g_o, delta_o, nu_o, bH_o, bD_o, alpha_o,
        case_o)

    if not store:
        return Trajectory(
            t=np.array([]), xi=np.array([]), eps=np.array([]),
            eps_sb=np.array([]), u=np.zeros((0, m)), gamma=np.zeros((0, p)),
            delta=np.array([]), nu=np.array([]),
            betaH=np.zeros((0, m)), betaD=np.zeros((0, p)), alpha=np.array([]),
            case=[], max_neg=float(max_neg), confined=bool(confined))

    if single:
        case = [_CASE_SINGLE[int(c)] for c in case_o]
    else:
        case = ["V" if c == -1 else ("I" if c == 0 else f"sat({int(c)})")
                for c in case_o]
    return Trajectory(
        t=np.arange(n_steps) * cfg.dt, xi=xi_o, eps=eps_o, eps_sb=epssb_o,
        u=u_o, gamma=g_o, delta=delta_o, nu=nu_o, betaH=bH_o, betaD=bD_o,
        alpha=alpha_o, case=case, max_neg=float(max_neg),
        confined=bool(confined))


def _run_trajectory_python(sys: System, cfg: SimConfig, rho0: np.ndarray,
                           seed: int = 0, store: bool = True,
                           design_sys: System | None = None,
                           t_stop: float | None = None) -> Trajectory:
    """Integrate one closed-loop sample path.

    Parameters
    ----------
    sys        : the true plant.  Governs the Milstein propagation and the
                 physicality monitor.
    cfg        : closed-loop configuration.
    rho0       : initial density operator.
    seed       : seed for the per-trajectory Wiener increments.
    store      : if False, only the scalar diagnostics (``confined``,
                 ``max_neg``) are kept and the per-step arrays are left
                 empty -- cheaper for large Monte Carlo ensembles.
    design_sys : optional controller model.  When given, the QP coefficients
                 (mu, beta^H, beta^D, sigma) are evaluated on ``design_sys``
                 instead of ``sys``, while the state is still propagated by
                 the true ``sys``.  This is the robust / mismatched
                 controller of Proposition IV.3: set ``design_sys`` to the
                 worst-case vertex of the uncertainty set and ``sys`` to the
                 (unknown) true plant.  When ``None`` the controller is
                 nominal (design == plant).  ``design_sys`` must share the
                 dimension and channel counts of ``sys``.
    """
    ctrl_sys = sys if design_sys is None else design_sys
    if ctrl_sys is not sys:
        if (ctrl_sys.N, ctrl_sys.m, ctrl_sys.p) != (sys.N, sys.m, sys.p):
            raise ValueError("design_sys must match the plant dimension "
                             "and channel counts")

    rng = np.random.default_rng(seed)
    barrier = FunnelGaugeBarrier()
    fun = cfg.funnel
    n_steps = int(round((fun.T if t_stop is None else t_stop) / cfg.dt))
    m, p = sys.m, sys.p

    x = to_bloch(sys, rho0)
    umax = np.full(m, cfg.umax)
    gmax = np.full(p, cfg.gmax)

    rec_t, rec_xi, rec_eps, rec_epssb = [], [], [], []
    rec_u, rec_g, rec_d, rec_nu, rec_bH, rec_case = [], [], [], [], [], []
    rec_bD, rec_alpha = [], []
    max_neg = 0.0
    confined = True

    single = (m == 1 and p == 1)

    for n in range(n_steps):
        t = n * cfg.dt
        rho = from_bloch(sys, x)

        # monitor physicality (Algorithm 1 reconstructs without projection)
        ev = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        max_neg = max(max_neg, float(-ev.min()))

        xi = infidelity(sys, rho)
        eps_t = float(fun.eps(t))
        if xi >= eps_t:
            confined = False
        sb_t = cfg.theta_b * eps_t               # relative buffer width
        s = max(eps_t - xi, sb_t)
        V = float(barrier.V(s, eps_t))
        kappa_V = float(barrier.kappa_V(s))

        # controller coefficients -- evaluated on the design model, which
        # equals the plant unless a robust/mismatched design is requested
        co = coefficients_generic(ctrl_sys, rho)
        betaH = np.atleast_1d(co["betaH"]).astype(float)
        betaD = np.atleast_1d(co["betaD"]).astype(float)
        sigma = float(co["sigma"])
        mu = float(co["mu"])

        # gauge alpha: the contraction charge carries the factor xi/eps,
        # vanishing at the target (revised eq. for alpha)
        alpha = (mu - (xi / eps_t) * float(fun.eps_dot(t))
                 + 0.5 * kappa_V * sigma ** 2)

        wu = _weights(cfg, betaH)
        qp = QPData(alpha=alpha, betaH=betaH, betaD=betaD, V=V, lam=cfg.lam,
                    wu=wu,
                    wgamma=np.full(p, cfg.wgamma), wdelta=cfg.wdelta,
                    umax=umax, gmax=gmax)
        res = solve_closed_form(qp) if single else solve_multichannel(qp)

        if store:
            rec_t.append(t); rec_xi.append(xi)
            rec_eps.append(eps_t); rec_epssb.append(eps_t - sb_t)
            rec_u.append(res.u.copy()); rec_g.append(res.gamma.copy())
            rec_d.append(res.delta); rec_nu.append(res.nu)
            rec_bH.append(betaH.copy()); rec_case.append(res.case)
            rec_bD.append(betaD.copy()); rec_alpha.append(alpha)

        dW = rng.normal(0.0, np.sqrt(cfg.dt))
        x = milstein_step(sys, x, res.u, res.gamma, cfg.dt, dW)
        if cfg.project:
            x, neg = project_physical(sys, x)
            max_neg = max(max_neg, neg)

    return Trajectory(
        t=np.array(rec_t), xi=np.array(rec_xi),
        eps=np.array(rec_eps), eps_sb=np.array(rec_epssb),
        u=np.array(rec_u) if rec_u else np.zeros((0, m)),
        gamma=np.array(rec_g) if rec_g else np.zeros((0, p)),
        delta=np.array(rec_d), nu=np.array(rec_nu),
        betaH=np.array(rec_bH) if rec_bH else np.zeros((0, m)),
        betaD=np.array(rec_bD) if rec_bD else np.zeros((0, p)),
        alpha=np.array(rec_alpha),
        case=rec_case, max_neg=max_neg, confined=confined,
    )


def run_ensemble(sys: System, cfg: SimConfig, rho0: np.ndarray,
                 M: int, seed0: int = 0, design_sys: System | None = None):
    """Run M independent trajectories; return list and confinement count.

    ``design_sys`` is forwarded to ``run_trajectory`` unchanged, so a robust
    or parametrically-mismatched controller can be Monte-Carlo'd directly.
    """
    trajs = [run_trajectory(sys, cfg, rho0, seed=seed0 + k, store=(k < 60),
                            design_sys=design_sys)
             for k in range(M)]
    n_conf = sum(tr.confined for tr in trajs)
    return trajs, n_conf
