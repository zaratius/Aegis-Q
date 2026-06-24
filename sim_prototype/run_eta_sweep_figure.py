"""
figures.py -- publication figures for the QDBC numerical study (Section V-E).

Currently provides the consolidated breach-refinement figure that replaces the
old fig_breach_refinement.pdf:

  Panel (a)  exit rate vs measurement efficiency eta  -- the noise lever.
             Both the shell-exit rate P[tau_Omega<=T] and the funnel-exit
             (true-breach) rate P[tau_aleph<=T] fall monotonically to zero as
             eta->0; at eta=0 the dynamics are deterministic and both vanish.
             This is the affirmative evidence for Remark IV.1: the exit rate is
             the Doob-bound content set by the measurement strength.

  Panel (b)  exit rate vs integration step dt  -- the discretization control.
             Both rates are dt-stable across the refinement: they plateau,
             departing from the O(dt) reference an artifact would track. The
             exits are continuous-time, not numerical.

Data is produced by eta_sweep.py (eta_sweep_results.json). The dt=DT_REPORT
point of panel (b) is the eta=0.6 baseline of panel (a) -- the two panels share
that condition, so it is not recomputed.

Usage:
  python figures.py [results.json] [out.pdf]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "eta_sweep_results.json")
OUT = os.path.join(HERE, "figures", "fig_breach_refinement.pdf")

DT_REPORT = 1.5e-4  # the reported / converged step (panel-b baseline)

# print-friendly palette; funnel (safety-critical) carries the attention color
C_SHELL = "#2a9d8f"   # teal  -- shell-exit, tau_Omega (design-margin erosion)
C_FUNNEL = "#c1272d"  # red   -- funnel-exit, tau_aleph (true breach)
C_REF = "#9a9a9a"     # grey  -- O(dt) artifact reference


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.fontsize": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def _rate(rec):
    """(value, n, p_shell, lo_s, hi_s, p_funnel, lo_f, hi_f) in percent."""
    n = len(rec["seeds"])
    ks, kf = sum(rec["shell"]), sum(rec["funnel"])
    ls, hs = clopper_pearson(ks, n)
    lf, hf = clopper_pearson(kf, n)
    return (rec["value"], n, 100 * ks / n, 100 * ls, 100 * hs,
            100 * kf / n, 100 * lf, 100 * hf)


def _series(db, knob):
    pts = sorted((_rate(r) for lbl, r in db.items() if r["knob"] == knob),
                 key=lambda t: t[0])
    return list(zip(*pts)) if pts else None


def _errbars(ax, x, p, lo, hi, color, marker, label, logx=False):
    p, lo, hi = np.asarray(p), np.asarray(lo), np.asarray(hi)
    yerr = np.vstack([p - lo, hi - p])
    ax.errorbar(x, p, yerr=yerr, fmt=marker, color=color, ecolor=color,
                ms=4, mfc=color, mec=color, elinewidth=0.9, capsize=2.2,
                capthick=0.9, label=label, zorder=3)


def fig_breach_refinement(db, out=OUT):
    _style()
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(3.5, 4.5), sharey=True,
        gridspec_kw=dict(hspace=0.42))

    # ---- panel (a): vs eta ------------------------------------------------ #
    eta = _series(db, "eta")
    val, n, ps, ls, hs, pf, lf, hf = (list(c) for c in eta)
    # eta=0 is deterministic (sigma_xi=0): the outcome is exact, not sampled,
    # so it carries no statistical uncertainty -- suppress its error bar.
    for i, v in enumerate(val):
        if v == 0.0:
            ls[i] = hs[i] = ps[i]
            lf[i] = hf[i] = pf[i]
    _errbars(axA, val, ps, ls, hs, C_SHELL, "o", r"shell-exit $\tau_\Omega$")
    _errbars(axA, val, pf, lf, hf, C_FUNNEL, "s", r"funnel-exit $\tau_\aleph$")
    axA.axvline(0.0, color="k", lw=0.5, ls=":", alpha=0.5)
    axA.annotate("deterministic\n(noise off)", xy=(0.0, 0.0),
                 xytext=(0.085, 6.2), fontsize=6.2, color="0.35",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
    axA.set_xlim(-0.03, 0.66)
    axA.set_xlabel(r"measurement efficiency $\eta$")
    axA.set_ylabel(r"exit rate $P[\tau_\bullet \leq T]$  (%)")
    axA.set_title(r"(a) noise lever: rate $\to 0$ as $\eta\to 0$",
                  fontsize=7.5)
    axA.legend(loc="upper left")

    # ---- panel (b): vs dt ------------------------------------------------- #
    dt = _series(db, "dt")
    if dt:
        dval, dn, dps, dls, dhs, dpf, dlf, dhf = (list(c) for c in dt)
    else:
        dval = []
    # inject the reported-dt baseline from the eta=0.6 condition (shared point)
    base = db.get("eta=0.6")
    if base is not None and DT_REPORT not in dval:
        v, nn, bps, bls, bhs, bpf, blf, bhf = _rate(base)
        dval.append(DT_REPORT)
        dps.append(bps); dls.append(bls); dhs.append(bhs)
        dpf.append(bpf); dlf.append(blf); dhf.append(bhf)
        order = np.argsort(dval)
        dval = list(np.array(dval)[order])
        dps = list(np.array(dps)[order]); dls = list(np.array(dls)[order])
        dhs = list(np.array(dhs)[order]); dpf = list(np.array(dpf)[order])
        dlf = list(np.array(dlf)[order]); dhf = list(np.array(dhf)[order])

    dval = np.array(dval)
    _errbars(axB, dval, dps, dls, dhs, C_SHELL, "o",
             r"shell-exit $\tau_\Omega$", logx=True)
    _errbars(axB, dval, dpf, dlf, dhf, C_FUNNEL, "s",
             r"funnel-exit $\tau_\aleph$", logx=True)

    # O(dt) reference anchored at the COARSEST dt, per series (what an artifact
    # would follow); the data sits well above it.
    dt_coarse = dval.max()
    grid = np.array(sorted(dval))
    refS = np.array(dps)[np.argmax(dval)] * grid / dt_coarse
    refF = np.array(dpf)[np.argmax(dval)] * grid / dt_coarse
    axB.plot(grid, refS, ls="--", lw=0.9, color=C_REF, alpha=0.8, zorder=1)
    axB.plot(grid, refF, ls="--", lw=0.9, color=C_REF, alpha=0.8, zorder=1)
    axB.plot([], [], ls="--", lw=0.9, color=C_REF,
             label=r"$\mathcal{O}(\Delta t)$ (artifact)")

    axB.set_xscale("log")
    axB.invert_xaxis()  # refinement: coarse -> fine, left -> right
    axB.set_xlabel(r"integration step $\Delta t$  ($\times 10^{-4}$)")
    axB.set_title(r"(b) refinement: both rates $\Delta t$-stable",
                  fontsize=7.5)
    axB.set_xticks(grid)
    axB.set_xticklabels([f"{v*1e4:g}" for v in grid])
    axB.minorticks_off()
    axB.set_ylabel(r"exit rate $P[\tau_\bullet \leq T]$  (%)")
    axB.legend(loc="lower left")

    axA.set_ylim(-1.5, 30)
    for ax in (axA, axB):
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)

    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RESULTS
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(path) as f:
        db = json.load(f)
    p = fig_breach_refinement(db, out)
    print("wrote", p)


if __name__ == "__main__":
    main()