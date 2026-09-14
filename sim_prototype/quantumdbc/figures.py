"""
Figure generation for Section V-E (Bell-state instance).

Two figures are produced:

* ``fig_bell_chatter``  -- the E2 comparison: the coherent control u(t)
  under the unregularized law (w_u = 1) and the regularized law,
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
matplotlib.use("pgf")
_PGF_PREAMBLE = "\n".join([
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amsmath,amssymb,mathtools}",
    r"\usepackage{bm}",
])

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science'/'ieee'/... styles)

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
    plt.style.use(["science", "ieee", "grid"])
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": _PGF_PREAMBLE,
        "text.usetex": False,            # pgf renders through pdflatex itself
        "axes.grid.axis": "y",
        "grid.color": "#E9E9E9", "grid.linewidth": 0.4, "grid.alpha": 0.6,
        "legend.frameon": False,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "savefig.dpi": 600, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# manuscript palette -- sampled from the Section V figures
_C_UNREG = "#BF0606"      # unregularized / first series -- red
_C_REG = "#000000"        # regularized   / second series -- teal
_C_BLUE = "#00629b"       # tertiary series -- blue (E1/E3 ensembles)
_C_AUX = "#8A8A8A"        # auxiliary lines (funnel, references) -- grey

# rate-axis label: time is in units of the measurement rate Gamma_m = 1
_T_LABEL = r"time  $t$ ($\Gamma_m^{-1}$)"
_DT_LABEL = r"integration step  $\Delta t$ ($10^{-4}\,\Gamma_m^{-1}$)"


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

    common = dict(funnel=funnel, dt=dt, lam=0.5, theta_b=0.30,
                  wgamma=1.0, wdelta=1e3)
    cfg_un = SimConfig(regularized=False, **common)
    cfg_re = SimConfig(regularized=True, c=20.0, **common)

    tr_un = run_trajectory(sys_, cfg_un, rho0, seed=seed)
    tr_re = run_trajectory(sys_, cfg_re, rho0, seed=seed)   # SAME seed

    fig, ax = plt.subplots(3, 1, figsize=(5.2, 4.2), sharex=True)

    # panel 1: coherent control u_1(t)
    ax[0].plot(tr_un.t, tr_un.u[:, 0], color=_C_UNREG, lw=0.6,
               label="unregularized")
    ax[0].plot(tr_re.t, tr_re.u[:, 0], color=_C_REG, lw=1.4, ls="--",
               label="regularized")
    ax[0].set_ylabel(r"control  $u_1(t)$")
    ax[0].set_ylim(-1.25, 1.25)
    ax[0].legend(loc="upper left", frameon=False, fontsize=8)
    ax[0].axhline(0, color=_C_AUX, lw=0.4)

    # --- magnification inset: resolve the sign-flips into a sawtooth -----
    # Pick a short window after the chattering onset.  The onset is where
    # the unregularized control first saturates; take a window a little
    # past it so the inset shows steady bang-bang switching, and make it
    # narrow enough (~25 steps) that individual flips are visible.
    u_un = tr_un.u[:, 0]
    sat = np.where(np.abs(u_un) > 0.5)[0]
    onset = int(sat[0]) if sat.size else len(u_un) // 3
    z0 = min(onset + 40, len(u_un) - 30)
    z1 = min(z0 + 25, len(u_un))
    tz = tr_un.t[z0:z1]

    axins = ax[0].inset_axes([0.62, 0.10, 0.34, 0.62])
    axins.plot(tz, u_un[z0:z1], color=_C_UNREG, lw=0.8,
               marker="o", ms=2.5, mfc=_C_UNREG, mec=_C_UNREG)
    axins.plot(tr_re.t[z0:z1], tr_re.u[z0:z1, 0], color=_C_REG,
               lw=1.4, ls="--")
    axins.axhline(0, color=_C_AUX, lw=0.4)
    axins.set_ylim(-1.25, 1.25)
    axins.set_xticklabels([])
    axins.set_yticks([-1, 0, 1])
    axins.tick_params(labelsize=6, length=2)
    axins.set_title("zoom", fontsize=7, pad=2)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    # box the source region on the main axes and connect it to the inset
    ax[0].indicate_inset_zoom(axins, edgecolor=_C_AUX, lw=0.5, alpha=0.6)

    # panel 2: conditional infidelity xi(t) with the funnel
    ax[1].plot(tr_un.t, tr_un.eps, color=_C_AUX, lw=0.8, ls="--",
               label=r"funnel $\epsilon(t)$")
    ax[1].plot(tr_un.t, tr_un.xi, color=_C_UNREG, lw=1.0,
               label=r"$\xi(t)$, unregularized")
    ax[1].plot(tr_re.t, tr_re.xi, color=_C_REG, lw=1.4, ls="--",
               label=r"$\xi(t)$, regularized")
    ax[1].set_ylabel(r"infidelity  $\xi(t)$")
    ax[1].legend(loc="upper right", frameon=False, fontsize=8)

    # panel 3: coherent gain magnitude |beta^H_xi(t)|
    ax[2].plot(tr_un.t, np.abs(tr_un.betaH[:, 0]), color=_C_UNREG, lw=0.8,
               label="unregularized")
    ax[2].plot(tr_re.t, np.abs(tr_re.betaH[:, 0]), color=_C_REG, lw=0.8,
               label="regularized")
    ax[2].set_ylabel(r"coherent gain  $|\beta^H_\xi|$")
    ax[2].set_xlabel(_T_LABEL)
    # legend in the clear pre-chattering region (t < 0.85); upper right sits
    # on the saturated band and is illegible in print
    ax[2].legend(loc="upper left", frameon=False, fontsize=8)

    fig.align_ylabels(ax)
    # figure title carried by the LaTeX caption
    path = os.path.join(outdir, "fig_bell_chatter.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# control experiment -- chattering is Delta t-robust, not an artifact
# --------------------------------------------------------------------------
def fig_qubit_confinement(outdir: str, trajs: list, eps_curve,
                          t_curve, ci=None, sb_curve=None,
                          bound=None) -> str:
    """E1 figure: qubit funnel confinement and Monte Carlo summary.

    Parameters
    ----------
    trajs    : list of Trajectory (a subset is plotted individually;
               breaching paths are drawn first so the rare event is visible)
    eps_curve, t_curve : the funnel epsilon(t) and its time grid
    ci       : optional (point, lo, hi) confinement fraction; drawn as a
               "confined: ... (95% CI ...)" line at bottom right
    sb_curve : optional shell edge (1-theta_b) epsilon(t); drawn dashed
    bound    : optional certified shell-exit bound; annotated as
               P[tau_Omega <= T] <= bound below the CI line
    """
    _style()
    n_show = min(25, len(trajs))
    M = len(trajs)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    # xi(t) sample paths. The drawn subset holds the ensemble's proportions
    # in spirit: at most 5 breaching paths (so the rare event is visible
    # without repainting a mostly-confined ensemble red), the rest confined.
    breach_idx = [i for i in range(M) if not trajs[i].confined][:5]
    conf_idx = [i for i in range(M) if trajs[i].confined]
    subset = breach_idx + conf_idx[:n_show - len(breach_idx)]
    for i in subset:
        tr = trajs[i]
        breached = not tr.confined
        ax.plot(tr.t, tr.xi,
               color=(_C_UNREG if breached else _C_AUX),
               lw=(0.7 if breached else 0.4),
               alpha=(0.9 if breached else 0.55),
               zorder=(3 if breached else 1))
    ax.plot(t_curve, eps_curve, color="k", lw=1.6,
           label=r"funnel $\epsilon(t)$")
    if sb_curve is not None:
        ax.plot(t_curve, sb_curve, color="k", lw=0.9, ls="--",
               label=r"shell edge $(1-\theta_b)\,\epsilon(t)$")
    ax.plot([], [], color=_C_AUX, lw=0.8, label=r"$\xi(t)$, confined")
    ax.plot([], [], color=_C_UNREG, lw=0.8,
           label=r"$\xi(t)$, breaching")
    ax.set_ylabel(r"infidelity  $\xi$")
    ax.set_xlabel(_T_LABEL)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    lines = []
    if ci is not None:
        pt, lo, hi = ci
        lines.append(f"confined: {pt*100:.1f}%  "
                     f"(95% CI [{lo*100:.1f}, {hi*100:.1f}])")
    if bound is not None:
        lines.append(rf"$\mathbb{{P}}[\tau_\Omega\le T]\le{bound:.3f}$")
    if lines:
        # anchored above the funnel tail so neither curve is occluded
        ax.text(0.98, 0.18, "\n".join(lines), transform=ax.transAxes,
                fontsize=7.5, color=_C_AUX, ha="right", va="bottom",
                zorder=5, bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.85, pad=1.5))

    fig.tight_layout()
    path = os.path.join(outdir, "fig_qubit_confinement.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_bell_confinement(outdir: str, t, eps_curve, sb_curve,
                         xi_paths, funnel_breached, ci=None,
                         bound=None, u_paths=None, bH_paths=None) -> str:
    """Certified Bell MC figure: ensemble confinement, parallel to
    fig_qubit_confinement, optionally completed with the applied coherent
    control and the coherent gain along the same sample paths.

    Parameters
    ----------
    t            : shared time grid (n_steps,)
    eps_curve    : funnel epsilon(t) on that grid
    sb_curve     : shell edge (1-theta_b) epsilon(t) on that grid
    xi_paths     : list/array of xi(t) sample paths (all M, used for the
                   pointwise frequency; a subset is drawn individually)
    funnel_breached : boolean per path, True if xi >= eps ever
    ci           : optional (point, lo, hi) funnel-confinement fraction
    bound        : optional certified shell-exit bound to annotate
    u_paths      : optional per-path u_1(t) traces (same length as
                   xi_paths); adds a control panel
    bH_paths     : optional per-path |beta^H_1(t)| traces; adds a gain panel
    """
    _style()
    xi_paths = [np.asarray(x, dtype=float) for x in xi_paths]
    M = len(xi_paths)
    n_show = min(25, M)
    extra = u_paths is not None and bH_paths is not None

    # the pgf backend emits every vertex as TeX tokens; at T/dt = 16000
    # steps x 25 paths x 3 spaghetti panels TeX's memory overflows. Drawn
    # curves are decimated to ~2000 vertices (far beyond print resolution);
    # all statistics (confinement frequency) are computed at full
    # resolution before decimation.
    stride = max(1, len(np.asarray(t)) // 2000)
    td = np.asarray(t, dtype=float)[::stride]

    def _dec(x):
        return np.asarray(x, dtype=float)[::stride]

    n_rows = 3 if extra else 1
    heights = [2, 1, 1] if extra else [1]
    fig, ax = plt.subplots(n_rows, 1,
                           figsize=(5.4, 6.5 if extra else 3.8),
                           gridspec_kw={"height_ratios": heights},
                           sharex=True, squeeze=False)
    ax = ax[:, 0]

    # drawn subset: at most 5 breaching paths (rare event visible without
    # repainting a mostly-confined ensemble red), the rest confined
    breach_idx = [i for i in range(M) if funnel_breached[i]][:5]
    conf_idx = [i for i in range(M) if not funnel_breached[i]]
    chosen = breach_idx + conf_idx[:n_show - len(breach_idx)]

    def _pathstyle(breached):
        return dict(color=(_C_UNREG if breached else _C_AUX),
                    lw=(0.7 if breached else 0.4),
                    alpha=(0.9 if breached else 0.55),
                    zorder=(3 if breached else 1))

    for i in chosen:
        ax[0].plot(td, _dec(xi_paths[i]), **_pathstyle(funnel_breached[i]))
    ax[0].plot(td, _dec(eps_curve), color="k", lw=1.6,
               label=r"funnel $\epsilon(t)$")
    ax[0].plot(td, _dec(sb_curve), color="k", lw=0.9, ls="--",
               label=r"shell edge $(1-\theta_b)\,\epsilon(t)$")
    ax[0].plot([], [], color=_C_AUX, lw=0.8, label=r"$\xi(t)$, confined")
    ax[0].plot([], [], color=_C_UNREG, lw=0.8,
               label=r"$\xi(t)$, breaching")
    ax[0].set_ylabel(r"infidelity  $\xi$")
    ax[0].set_ylim(0, None)
    ax[0].legend(loc="upper right", frameon=False, fontsize=8)

    lines = []
    if ci is not None:
        pt, lo, hi = ci
        lines.append(f"confined: {pt*100:.1f}%  "
                     f"(95% CI [{lo*100:.1f}, {hi*100:.1f}])")
    if bound is not None:
        lines.append(rf"$\mathbb{{P}}[\tau_\Omega\le T]\le{bound:.3f}$")
    if lines:
        # anchored above the funnel tail so neither curve is occluded
        ax[0].text(0.98, 0.28, "\n".join(lines), transform=ax[0].transAxes,
                   fontsize=7.5, color=_C_AUX, ha="right", va="bottom",
                   zorder=5, bbox=dict(facecolor="white", edgecolor="none",
                                       alpha=0.85, pad=1.5))

    if extra:
        # same subset, same colors: the applied coherent control and the
        # coherent gain the law responds to (first channel; the
        # preparation tilt sits on qubit 1)
        for i in chosen:
            ax[1].plot(td, _dec(u_paths[i]),
                       **_pathstyle(funnel_breached[i]))
        ax[1].axhline(0.0, color=_C_AUX, lw=0.5, ls=":")
        ax[1].set_ylabel(r"control  $u_1$")
        for i in chosen:
            ax[2].plot(td, _dec(bH_paths[i]),
                       **_pathstyle(funnel_breached[i]))
        ax[2].set_ylabel(r"gain  $|\beta^{H_1}_\xi|$")
        ax[2].set_ylim(0, None)

    ax[-1].set_xlabel(_T_LABEL)
    fig.align_ylabels(ax)
    path = os.path.join(outdir, "fig_bell_confinement.pdf")
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
        cq = SimConfig(funnel=fq, dt=dt, lam=0.6, theta_b=0.10, regularized=False)
        tq = run_trajectory(sq, cq, rq, seed=seed)
        ff_q.append(np.mean(np.diff(np.sign(tq.u[:, 0])) != 0))

        cb = SimConfig(funnel=fb, dt=dt, lam=0.5, theta_b=0.10, regularized=False)
        tb = run_trajectory(sb, cb, rb, seed=seed)
        ff_b.append(np.mean(np.diff(np.sign(tb.u[:, 0])) != 0))

    dts_disp = np.asarray(dts) * 1e4   # display in units of 10^{-4} Gamma_m^{-1}
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.semilogx(dts_disp, ff_q, "o-", color=_C_BLUE, label="qubit")
    ax.semilogx(dts_disp, ff_b, "s-", color=_C_UNREG, label=r"Bell, $|00\rangle$")
    ax.set_xlabel(_DT_LABEL)
    ax.set_ylabel("control sign-flip fraction")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()       # Delta t -> 0 to the right
    ax.legend(frameon=False)
    # title carried by the LaTeX caption
    ax.annotate("a discretization artifact\nwould vanish here",
                xy=(dts_disp[-1], 0.2), xytext=(dts_disp[1], 0.35),
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

    Under the funnel gauge the speed limit at a state rho carries the
    prefactor eps/xi (revised Proposition III.2):

        Phi(rho,t) = (eps/xi) [ sum |beta^H_i| u_max + sum |beta^D_j| g_max
                                - mu - (1/2) kappa_V sigma^2 ],

    and the funnel is feasible at rho iff |eps_dot| <= Phi.  The stored
    gauge alpha is  mu - (xi/eps) eps_dot + (1/2) kappa_V sigma^2, so
    -mu - (1/2) kappa_V sigma^2 = -alpha - (xi/eps) eps_dot, and

        Phi = (eps/xi) ( sum|bH| u_max + sum|bD| g_max - alpha ) - eps_dot.

    The condition is only meaningful where the barrier is active, i.e. where
    the state is in the outer shell of the funnel,  xi > shell * eps.  Deep
    inside the funnel the actuation gains vanish (the state is at the
    target) and Phi < |eps_dot| there carries no information -- there is no
    confinement demand to meet.  The figure therefore evaluates Phi on the
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
        with np.errstate(divide="ignore", invalid="ignore"):
            pref = tr.eps / tr.xi                # gauge prefactor eps/xi
            V = pref * (bH * umax + bD * gmax - tr.alpha) - eps_dot
        in_shell = tr.xi > shell * tr.eps        # pref finite on the shell
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
def fig_breach_refinement(outdir, dts, shell_rates, ci_los, ci_his,
                          alias_dt=2.0e-4):
    """Shell-exit rate vs Delta t for the qubit ensemble.

    shell_rates / ci_los / ci_his are the SHELL-exit (s < theta_b eps) rate and its
    95% Clopper-Pearson interval, indexed over ``dts``.  Points with
    dt > alias_dt are drawn hollow: at Omega = 20 the closed loop aliases
    there (STATUS.md) and those points do not belong to the continuum trend.
    """
    # local style (kept self-contained; matches _style() palette). The pgf
    # preamble must be set here as well: this function does not go through
    # _style(), and without amssymb the \mathbb in the y-label halts the
    # pdflatex run at savefig time.
    _C_REG = "#1a9988"; _C_AUX = "#7f7f7f"
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.grid": True, "grid.color": "#d9d9d9",
                         "grid.linewidth": 0.5, "savefig.bbox": "tight",
                         "figure.dpi": 150,
                         "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                         "pgf.preamble": _PGF_PREAMBLE})
 
    dts = np.asarray(dts, float)
    rates = np.asarray(shell_rates, float)
    lo = np.asarray(ci_los, float)
    hi = np.asarray(ci_his, float)
 
    # O(dt) reference anchored at the FINEST un-aliased point, extended up
    smooth = dts <= alias_dt
    anchor = int(np.argmin(dts))           # finest dt
    ref = rates[anchor] * dts / dts[anchor]
 
    dts_disp = dts * 1e4                   # display in units of 10^{-4} Gamma_m^{-1}
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(dts_disp, ref, color=_C_AUX, lw=0.9, ls="--",
            label=r"$\mathcal{O}(\Delta t)$: what an artifact would follow")

    # solid markers for smooth points, hollow for aliased ones
    ax.errorbar(dts_disp[smooth], rates[smooth],
                yerr=[rates[smooth] - lo[smooth], hi[smooth] - rates[smooth]],
                fmt="o-", color=_C_REG, lw=1.0, capsize=3,
                label=r"shell-exit rate (95\% CI)")
    if (~smooth).any():
        ax.errorbar(dts_disp[~smooth], rates[~smooth],
                    yerr=[rates[~smooth] - lo[~smooth], hi[~smooth] - rates[~smooth]],
                    fmt="o", mfc="white", color=_C_REG, lw=1.0, capsize=3,
                    label=r"aliased ($\Omega\Delta t$ too large)")
 
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"integration step  $\Delta t$ ($10^{-4}\,\Gamma_m^{-1}$)")
    ax.set_ylabel(r"shell-exit rate  $\mathbb{P}[\tau_\Omega \leq T]$")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
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