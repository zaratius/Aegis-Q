"""
Closed-loop driver: Algorithm 1 of the manuscript.

Each step performs (i) coefficient evaluation, (ii) closed-form QP solution,
(iii) Milstein propagation.  The previous coherent control is carried as
additional scalar state, initialised to zero.

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
from .barrier import LogBarrier, ExpFunnel
from .coefficients import coefficients_generic, infidelity, to_bloch, from_bloch
from .controller import QPData, solve_closed_form, solve_multichannel
from .integrator import milstein_step, project_physical


@dataclass
class SimConfig:
    """Closed-loop configuration."""
    funnel: ExpFunnel
    lam: float = 1.0                 # exponential barrier-decay rate
    s_b: float = 0.05                # buffer
    c: float = 1.0                   # state-weight scale
    eps_f: float = 1e-2              # state-weight floor
    wr: float = 5.0                  # temporal weight
    wgamma: float = 1.0              # dissipative weight
    wdelta: float = 1e3              # slack penalty
    umax: float = 1.0                # coherent bound (per channel)
    gmax: float = 1.0                # dissipative bound (per channel)
    dt: float = 1e-3
    regularized: bool = True         # False -> w_r = 0, w_u == 1
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
        wr = np.full(m, cfg.wr)
    else:
        wu = np.ones(m)
        wr = np.zeros(m)
    return wu, wr


def run_trajectory(sys: System, cfg: SimConfig, rho0: np.ndarray,
                   seed: int = 0, store: bool = True,
                   design_sys: System | None = None) -> Trajectory:
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
    barrier = LogBarrier(s_bar=cfg.funnel.eps0)
    fun = cfg.funnel
    n_steps = int(round(fun.T / cfg.dt))
    m, p = sys.m, sys.p

    x = to_bloch(sys, rho0)
    u_prev = np.zeros(m)
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
        s = max(eps_t - xi, cfg.s_b)
        V = float(barrier.V(s))
        kappa_V = float(barrier.kappa_V(s))

        # controller coefficients -- evaluated on the design model, which
        # equals the plant unless a robust/mismatched design is requested
        co = coefficients_generic(ctrl_sys, rho)
        betaH = np.atleast_1d(co["betaH"]).astype(float)
        betaD = np.atleast_1d(co["betaD"]).astype(float)
        sigma = float(co["sigma"])
        mu = float(co["mu"])

        alpha = mu - float(fun.eps_dot(t)) + 0.5 * kappa_V * sigma ** 2

        wu, wr = _weights(cfg, betaH)
        qp = QPData(alpha=alpha, betaH=betaH, betaD=betaD, V=V, lam=cfg.lam,
                    u_prev=u_prev, wu=wu, wr=wr,
                    wgamma=np.full(p, cfg.wgamma), wdelta=cfg.wdelta,
                    umax=umax, gmax=gmax)
        res = solve_closed_form(qp) if single else solve_multichannel(qp)

        if store:
            rec_t.append(t); rec_xi.append(xi)
            rec_eps.append(eps_t); rec_epssb.append(eps_t - cfg.s_b)
            rec_u.append(res.u.copy()); rec_g.append(res.gamma.copy())
            rec_d.append(res.delta); rec_nu.append(res.nu)
            rec_bH.append(betaH.copy()); rec_case.append(res.case)
            rec_bD.append(betaD.copy()); rec_alpha.append(alpha)

        dW = rng.normal(0.0, np.sqrt(cfg.dt))
        x = milstein_step(sys, x, res.u, res.gamma, cfg.dt, dW)
        if cfg.project:
            x, neg = project_physical(sys, x)
            max_neg = max(max_neg, neg)
        u_prev = res.u

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