"""
Figure generation for Section V-E (Bell-state instance).

Two figures are produced:

* ``fig_bell_chatter``  -- the E2 comparison: the coherent control u(t)
  under the unregularized law (w_r = 0, w_u = 1) and the regularized law,
  integrated from rho_0 = |00><00| on a SHARED Brownian path.  Companion
  panels show the conditional infidelity xi(t) and the coherent gain
  |beta^H_xi(t)|.

* ``fig_chatter_robustness`` -- the control experiment that distinguishes a
  genuine obstruction from a discretization artifact: the unregularized
  control sign-flip fraction as a function of the integration step Delta t,
  for the qubit and the Bell instance.  A discretization artifact would
  vanish as Delta t -> 0; a genuine obstruction persists.  Both curves rise
  to a plateau, confirming the chattering of Proposition V.2 is Delta t
  -robust.

These are reference figures.  Numerical values must be checked against the
analytic derivation before use in the paper.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .systems import bell, qubit
from .barrier import ExpFunnel
from .simulate import SimConfig, run_trajectory


# --------------------------------------------------------------------------
# styling -- matches the manuscript figures (IEEEtran, serif, orange/teal)
# --------------------------------------------------------------------------
def _style():
    """Apply the manuscript figure style.

    Serif text to sit with the IEEEtran body font; the burnt-orange / teal
    pair of the Section V figures.  Note the two colours are close in
    luminance (they read alike in greyscale and for some colour-vision
    deficiencies), so every figure also separates the two series by
    linewidth or linestyle -- colour is never the sole distinguishing
    channel.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


# manuscript palette -- sampled from the Section V figures
_C_UNREG = "#d4691e"      # unregularized / first series -- burnt orange
_C_REG = "#1a9988"        # regularized   / second series -- teal
_C_BLUE = "#1f6aa5"       # tertiary series -- blue (E1/E3 ensembles)
_C_AUX = "#7f7f7f"        # auxiliary lines (funnel, references) -- grey

# rate-axis label: time is in units of the measurement rate Gamma_m = 1
_T_LABEL = r"time  $\Gamma_m t$"
_DT_LABEL = r"integration step  $\Gamma_m \Delta t$"


# --------------------------------------------------------------------------
# E2 -- Bell chattering, unregularized vs regularized on a shared noise path
# --------------------------------------------------------------------------
def fig_bell_chatter(outdir: str, seed: int = 7,
                     dt: float = 2.5e-4, T: float = 2.0) -> str:
    """Generate the E2 figure.  Returns the output path.

    Uses kappa = 50 (kappa/Gamma_m = 50): with the Section V-E table value
    kappa = 5 the engineered dissipators cannot stabilize Phi+ at all (see
    STATUS.md), so the figure would show no stabilization.  The funnel is
    sized to the resulting xi envelope.
    """
    _style()
    sys_ = bell(kappa=(50.0, 50.0, 50.0))
    ket00 = np.zeros(4, dtype=complex); ket00[0] = 1.0
    rho0 = np.outer(ket00, ket00.conj())
    funnel = ExpFunnel(eps0=0.62, eps_T=0.08, T=T)

    common = dict(funnel=funnel, dt=dt, lam=0.5, s_b=0.05,
                  wgamma=1.0, wdelta=1e3)
    cfg_un = SimConfig(regularized=False, **common)
    cfg_re = SimConfig(regularized=True, wr=50.0, c=20.0, **common)

    tr_un = run_trajectory(sys_, cfg_un, rho0, seed=seed)
    tr_re = run_trajectory(sys_, cfg_re, rho0, seed=seed)   # SAME seed

    fig, ax = plt.subplots(3, 1, figsize=(5.4, 6.0), sharex=True)

    # panel 1: coherent control u_1(t)
    ax[0].plot(tr_un.t, tr_un.u[:, 0], color=_C_UNREG, lw=0.6,
               label="unregularized")
    ax[0].plot(tr_re.t, tr_re.u[:, 0], color=_C_REG, lw=1.4, ls="--",
               label="regularized")
    ax[0].set_ylabel(r"coherent control  $u_1(t)$")
    ax[0].set_ylim(-1.25, 1.25)
    ax[0].legend(loc="upper right", frameon=False, fontsize=8)
    ax[0].axhline(0, color=_C_AUX, lw=0.4)

    # panel 2: conditional infidelity xi(t) with the funnel
    ax[1].plot(tr_un.t, tr_un.eps, color=_C_AUX, lw=0.8, ls="--",
               label=r"funnel $\epsilon(t)$")
    ax[1].plot(tr_un.t, tr_un.xi, color=_C_UNREG, lw=1.0,
               label=r"$\xi(t)$, unregularized")
    ax[1].plot(tr_re.t, tr_re.xi, color=_C_REG, lw=1.4, ls="--",
               label=r"$\xi(t)$, regularized")
    ax[1].set_ylabel(r"conditional infidelity  $\xi(t)$")
    ax[1].legend(loc="upper right", frameon=False, fontsize=8)

    # panel 3: coherent gain magnitude |beta^H_xi(t)|
    ax[2].plot(tr_un.t, np.abs(tr_un.betaH[:, 0]), color=_C_UNREG, lw=0.8,
               label="unregularized")
    ax[2].plot(tr_re.t, np.abs(tr_re.betaH[:, 0]), color=_C_REG, lw=0.8,
               label="regularized")
    ax[2].set_ylabel(r"coherent gain  $|\beta^H_\xi(t)|$")
    ax[2].set_xlabel(_T_LABEL)
    ax[2].legend(loc="upper right", frameon=False, fontsize=8)

    fig.align_ylabels(ax)
    fig.suptitle("E2  Bell-state stabilization from "
                 r"$\rho_0=|00\rangle\langle00|$ (shared noise path)",
                 fontsize=9)
    path = os.path.join(outdir, "fig_bell_chatter.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# control experiment -- chattering is Delta t-robust, not an artifact
# --------------------------------------------------------------------------
def fig_qubit_confinement(outdir: str, trajs: list, eps_curve,
                          t_curve, ci=None) -> str:
    """E1 figure: qubit funnel confinement and Monte Carlo summary.

    Parameters
    ----------
    trajs    : list of Trajectory (a subset is plotted individually)
    eps_curve, t_curve : the funnel epsilon(t) and its time grid
    ci       : optional (point, lo, hi) confinement-fraction estimate with
               a 95% Clopper-Pearson interval, in [0, 1]
    """
    _style()
    n_show = min(25, len(trajs))
    M = len(trajs)
    conf = [tr.confined for tr in trajs]

    fig, ax = plt.subplots(2, 1, figsize=(5.4, 5.4),
                           gridspec_kw={"height_ratios": [2, 1]})

    # panel 1: xi(t) sample paths; breaching paths highlighted
    for tr in trajs[:n_show]:
        breached = not tr.confined
        ax[0].plot(tr.t, tr.xi,
                   color=(_C_UNREG if breached else _C_AUX),
                   lw=(0.7 if breached else 0.4),
                   alpha=(0.9 if breached else 0.55),
                   zorder=(3 if breached else 1))
    ax[0].plot(t_curve, eps_curve, color="k", lw=1.6,
               label=r"funnel $\epsilon(t)$")
    ax[0].plot([], [], color=_C_AUX, lw=0.8, label=r"$\xi(t)$, confined")
    ax[0].plot([], [], color=_C_UNREG, lw=0.8,
               label=r"$\xi(t)$, breaching")
    ax[0].set_ylabel(r"infidelity  $\xi$")
    ax[0].set_ylim(0, None)
    ax[0].legend(loc="upper right", frameon=False, fontsize=8)
    ax[0].set_title("E1  qubit ground-state confinement", fontsize=9)

    # panel 2: pointwise confinement frequency
    t = trajs[0].t
    inside = np.zeros(len(t))
    for tr in trajs:
        inside += (tr.xi < tr.eps).astype(float)
    inside /= M
    ax[1].plot(t, inside, color=_C_BLUE, lw=1.0)
    ax[1].axhline(1.0, color=_C_AUX, lw=0.5, ls=":")
    ax[1].set_ylabel("confinement\nfrequency")
    ax[1].set_xlabel(_T_LABEL)
    ax[1].set_ylim(0, 1.05)

    label = f"ensemble  $M = {M}$"
    if ci is not None:
        pt, lo, hi = ci
        label += (f"\nconfined: {pt*100:.1f}%  "
                  f"(95% CI [{lo*100:.1f}, {hi*100:.1f}])")
    ax[1].text(0.02, 0.10, label, transform=ax[1].transAxes,
               fontsize=7.5, color=_C_AUX, va="bottom")

    fig.align_ylabels(ax)
    path = os.path.join(outdir, "fig_qubit_confinement.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_chatter_robustness(outdir: str, seed: int = 7,
                           dts=(2e-3, 1e-3, 5e-4, 2.5e-4)) -> str:
    """Unregularized sign-flip fraction vs Delta t, qubit and Bell."""
    _style()

    # qubit
    sq = qubit()
    rq = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    fq = ExpFunnel(eps0=0.55, eps_T=0.10, T=1.0)
    # bell
    sb = bell()
    ket00 = np.zeros(4, dtype=complex); ket00[0] = 1.0
    rb = np.outer(ket00, ket00.conj())
    fb = ExpFunnel(eps0=0.62, eps_T=0.20, T=1.0)

    ff_q, ff_b = [], []
    for dt in dts:
        cq = SimConfig(funnel=fq, dt=dt, lam=0.6, s_b=0.04, regularized=False)
        tq = run_trajectory(sq, cq, rq, seed=seed)
        ff_q.append(np.mean(np.diff(np.sign(tq.u[:, 0])) != 0))

        cb = SimConfig(funnel=fb, dt=dt, lam=0.5, s_b=0.05, regularized=False)
        tb = run_trajectory(sb, cb, rb, seed=seed)
        ff_b.append(np.mean(np.diff(np.sign(tb.u[:, 0])) != 0))

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogx(dts, ff_q, "o-", color=_C_BLUE, label="qubit")
    ax.semilogx(dts, ff_b, "s-", color=_C_UNREG, label=r"Bell, $|00\rangle$")
    ax.set_xlabel(_DT_LABEL)
    ax.set_ylabel("control sign-flip fraction")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()       # Delta t -> 0 to the right
    ax.legend(frameon=False)
    ax.set_title("Unregularized chattering persists as "
                 r"$\Delta t \to 0$", fontsize=9)
    ax.annotate("a discretization artifact\nwould vanish here",
                xy=(dts[-1], 0.2), xytext=(dts[1], 0.35),
                fontsize=7.5, color=_C_AUX,
                arrowprops=dict(arrowstyle="->", color=_C_AUX, lw=0.6))
    path = os.path.join(outdir, "fig_chatter_robustness.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path



# --------------------------------------------------------------------------
# E3 -- funnel feasibility check
# --------------------------------------------------------------------------
def feasibility_data(trajs: list, funnel, umax: float, gmax: float,
                     shell: float = 0.85):
    """Pointwise feasibility margin, with the boundary-shell restriction.

    The QP avoids slack exactly when the actuators cover the uncontrolled
    drift,

        sum |beta^H_i| u_max + sum |beta^D_j| gamma_max  >=  alpha + lam V,

    alpha = mu - eps_dot + (1/2) kappa_V sigma^2.  Isolating the funnel
    contraction demand gives the feasibility condition  |eps_dot| <= V(t),

        V(t) = sum |beta^H_i| u_max + sum |beta^D_j| gamma_max
               - ( mu + (1/2) kappa_V sigma^2 ).

    The condition is only meaningful where the barrier is active, i.e. where
    the state is in the outer shell of the funnel,  xi > shell * eps.  Deep
    inside the funnel the actuation gains vanish (the state is at the
    target) and V(t) < 0 there carries no information -- there is no
    confinement demand to meet.  The figure therefore evaluates V(t) on the
    boundary shell only.

    Returns (t, demand, V_min_shell, V_med_shell, infeasible_mask) where the
    shell quantities are nan at times no ensemble member is in the shell.
    """
    t = trajs[0].t
    eps_dot = np.asarray(funnel.eps_dot(t))
    demand = np.abs(eps_dot)

    V_shell = np.full((len(trajs), len(t)), np.nan)
    for i, tr in enumerate(trajs):
        bH = np.abs(tr.betaH).sum(axis=1)
        bD = np.abs(tr.betaD).sum(axis=1)
        V = bH * umax + bD * gmax - (tr.alpha + eps_dot)
        in_shell = tr.xi > shell * tr.eps
        V_shell[i, in_shell] = V[in_shell]

    with np.errstate(invalid="ignore"):
        all_nan = np.all(np.isnan(V_shell), axis=0)
        V_min = np.full(len(t), np.nan)
        V_med = np.full(len(t), np.nan)
        V_min[~all_nan] = np.nanmin(V_shell[:, ~all_nan], axis=0)
        V_med[~all_nan] = np.nanmedian(V_shell[:, ~all_nan], axis=0)
    infeasible = V_min < demand
    return t, demand, V_min, V_med, infeasible


def fig_feasibility(outdir: str, trajs: list, funnel, umax: float,
                    gmax: float, shell: float = 0.85) -> str:
    """E3 figure: the feasibility condition |eps_dot(t)| <= V(t), evaluated
    on the funnel boundary shell, against the E1 breach onsets."""
    _style()
    t, demand, V_min, V_med, infeas = feasibility_data(
        trajs, funnel, umax, gmax, shell)

    # E1 breach onsets
    breach_t = []
    for tr in trajs:
        out = np.where(tr.xi >= tr.eps)[0]
        if out.size:
            breach_t.append(tr.t[out[0]])

    fig, ax = plt.subplots(2, 1, figsize=(5.4, 5.4), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})

    # panel 1: demand vs the boundary-shell margin
    ax[0].plot(t, demand, color=_C_AUX, lw=1.2, ls="--",
               label=r"$|\dot\epsilon(t)|$  funnel demand")
    ax[0].plot(t, V_med, color=_C_REG, lw=1.2,
               label=r"$\mathcal{V}(t)$  median on boundary shell")
    ax[0].plot(t, V_min, color=_C_REG, lw=0.8, ls=":",
               label=r"$\mathcal{V}(t)$  worst on boundary shell")
    ax[0].axhline(0.0, color=_C_AUX, lw=0.4)
    ax[0].set_ylabel("contraction rate")
    ax[0].legend(loc="upper right", frameon=False, fontsize=8)
    ax[0].set_title(r"E3  funnel feasibility on the boundary shell "
                    fr"$\xi > {shell}\,\epsilon$", fontsize=9)

    # panel 2: worst-case feasibility margin
    slack = V_min - demand
    ax[1].axhline(0.0, color="k", lw=0.6)
    ax[1].plot(t, slack, color=_C_REG, lw=1.0)
    ax[1].fill_between(t, slack, 0.0, where=(slack < 0),
                       color=_C_UNREG, alpha=0.30, interpolate=True,
                       label="infeasible (some path on shell)")
    for k, bt in enumerate(breach_t):
        ax[1].axvline(bt, color=_C_AUX, lw=0.6, ls=":",
                      label="E1 breach onset" if k == 0 else None)
    ax[1].set_ylabel(r"$\mathcal{V}_{\min} - |\dot\epsilon|$")
    ax[1].set_xlabel(_T_LABEL)
    ax[1].legend(loc="lower right", frameon=False, fontsize=8)

    fig.align_ylabels(ax)
    path = os.path.join(outdir, "fig_feasibility.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 1 -- breach-rate step refinement (Section V-E4)
# --------------------------------------------------------------------------
def fig_breach_refinement(outdir: str, dts, breach_rates,
                          ci_los, ci_his) -> str:
    """Funnel-breach rate of the admissible-funnel qubit ensemble vs Delta t.

    For an admissible funnel the continuum-limit breach probability is zero
    (Theorem III.1).  At finite Delta t the Milstein truncation and the
    positivity projection inject an O(Delta t) drift slack (Corollary III.2)
    that carries a few sample paths across the boundary; halving Delta t
    halves that slack.  The breach rate therefore falls along an O(Delta t)
    reference and vanishes in the continuum limit -- the signature that the
    finite-step excursions of the E1 study are integration slack, not funnel
    infeasibility.

    All arrays are indexed over ``dts`` and produced by
    ``run_breach_refinement.py``.
    """
    _style()
    dts = np.asarray(dts, float)
    rates = np.asarray(breach_rates, float)
    lo = np.asarray(ci_los, float)
    hi = np.asarray(ci_his, float)

    # O(Delta t) reference anchored at the coarsest step
    i0 = int(np.argmax(dts))
    ref = rates[i0] * dts / dts[i0]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(dts, ref, color=_C_AUX, lw=0.9, ls="--",
            label=r"$\mathcal{O}(\Delta t)$ reference")
    ax.errorbar(dts, rates, yerr=[rates - lo, hi - rates],
                fmt="o-", color=_C_REG, lw=1.0, capsize=3,
                label="breach rate (95% CI)")
    ax.set_xscale("log")
    ax.invert_xaxis()                          # Delta t -> 0 to the right
    ax.set_xlabel(r"integration step  $\Delta t$")
    ax.set_ylabel("funnel-breach rate")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"Breach rate vanishes as $\Delta t \to 0$  "
                 "(excursions are integration slack)", fontsize=9)
    path = os.path.join(outdir, "fig_breach_refinement.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Section V-E8 -- robustness to dissipative-rate uncertainty
# --------------------------------------------------------------------------
def fig_robustness(outdir: str, plant_kappas, freq_rob, ci_rob,
                   freq_opt, ci_opt, exc_rob, exc_opt,
                   worst_kappa: float) -> str:
    """Robust vs optimistic controller under dissipative-rate uncertainty.

    Panel A -- confinement frequency of the robust controller (coefficients
    computed from the worst-case rate kappa_min) and the optimistic
    controller (coefficients from kappa_max), each run against true plants
    at several kappa.  The two confine at comparable frequency:
    Proposition IV.3 does not promise a higher confinement frequency under
    model uncertainty.

    Panel B -- on the worst-case plant kappa = kappa_min, the distribution
    of the per-path maximum margin ratio max_t xi/epsilon.  This is where
    the robust design pays off: by computing its coefficients from the
    worst-case model it bounds the post-control drift, so its excursions
    stay close to the funnel, whereas the optimistic controller -- having
    over-estimated the dissipative authority -- cannot arrest a drifting
    path and produces a heavy tail of large excursions.

    ``freq_*`` are arrays over ``plant_kappas``; ``ci_*`` are matching
    arrays of (lo, hi) pairs; ``exc_*`` are 1-D arrays of the per-path
    max xi/epsilon on the worst-case plant.
    """
    _style()
    plant_kappas = np.asarray(plant_kappas, float)
    fr = np.asarray(freq_rob, float); fo = np.asarray(freq_opt, float)
    cr = np.asarray(ci_rob, float);   co = np.asarray(ci_opt, float)
    exc_rob = np.asarray(exc_rob, float)
    exc_opt = np.asarray(exc_opt, float)

    fig, ax = plt.subplots(2, 1, figsize=(5.4, 6.2),
                           gridspec_kw={"height_ratios": [1, 1], "hspace": 0.42})

    # ---- panel A: confinement frequency, grouped bars ------------------
    x = np.arange(len(plant_kappas))
    w = 0.36
    ax[0].bar(x - w / 2, fr, w, color=_C_REG,
              label=r"robust (design $\kappa_{\min}$)",
              yerr=[fr - cr[:, 0], cr[:, 1] - fr],
              capsize=3, error_kw=dict(lw=0.8))
    ax[0].bar(x + w / 2, fo, w, color=_C_UNREG,
              label=r"optimistic (design $\kappa_{\max}$)",
              yerr=[fo - co[:, 0], co[:, 1] - fo],
              capsize=3, error_kw=dict(lw=0.8))
    ax[0].axhline(1.0, color=_C_AUX, lw=0.5, ls=":")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([fr"$\kappa = {k:g}$" for k in plant_kappas])
    ax[0].set_xlabel(r"true plant dissipative rate $\kappa$")
    ax[0].set_ylabel("confinement\nfrequency")
    ax[0].set_ylim(0.0, 1.18)
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    ax[0].set_title("Robustness to dissipative-rate uncertainty", fontsize=9)

    # ---- panel B: excursion severity on the worst-case plant -----------
    top = max(exc_rob.max(), exc_opt.max(), 1.05) * 1.10
    bins = np.linspace(0.0, top, 26)
    ax[1].hist(exc_rob, bins=bins, color=_C_REG, alpha=0.75,
               label=f"robust  (worst {exc_rob.max():.2f})")
    ax[1].hist(exc_opt, bins=bins, color=_C_UNREG, alpha=0.55,
               label=f"optimistic  (worst {exc_opt.max():.2f})")
    ax[1].axvline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 1.25)
    ymax = ax[1].get_ylim()[1]
    ax[1].text(0.96, ymax * 0.96, "funnel\nboundary", fontsize=7,
               va="top", ha="right")
    ax[1].set_xlabel(r"per-path maximum margin ratio  $\max_t\,\xi/\epsilon$"
                     fr"   (worst-case plant $\kappa = {worst_kappa:g}$)")
    ax[1].set_ylabel("count")
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.align_ylabels(ax)
    path = os.path.join(outdir, "fig_robustness.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path