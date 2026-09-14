"""
Robustness to dissipative-rate uncertainty

    python scripts/run_robustness_acc.py [M]          # M defaults to 40
"""


# APPLE ACCELERATE GRIDLOCK PREVENTION 
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import concurrent.futures
from functools import partial, lru_cache

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, SimConfig, run_trajectory
from quantumdbc.study_config import (
    QUBIT_FUNNEL, QUBIT_THETA_B, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)
from quantumdbc.exit_metrics import exit_metrics


ROBUST_DT = 5e-4
SEED0 = 5000


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


@lru_cache(maxsize=8)
def _qubit_cached(kappa: float):
    return qubit(kappa=kappa)


def _robust_single(task, cfg, rho0, seed0):
    plant_kappa, design_kappa, design_label, k = task
    plant = _qubit_cached(plant_kappa)
    design = _qubit_cached(design_kappa)
    tr = run_trajectory(plant, cfg, rho0, seed=seed0 + k,
                        store=True, design_sys=design)
    m = exit_metrics(tr, cfg.theta_b)
    return (plant_kappa, design_label, bool(not m["funnel_exit"]),
            float(m["runmax"]))


def _plot_robustness_cdf(outdir, exc_rob, exc_opt):
    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the styles)
    plt.style.use(["science", "ieee", "grid"])
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "text.usetex": False, "savefig.dpi": 600, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.grid.axis": "y",
        "grid.color": "#E9E9E9", "grid.linewidth": 0.4, "grid.alpha": 0.6,
        "legend.frameon": False,
    })

    er = np.sort(np.asarray(exc_rob, float))
    eo = np.sort(np.asarray(exc_opt, float))
    yr = np.arange(1, len(er) + 1) / len(er)
    yo = np.arange(1, len(eo) + 1) / len(eo)
    wr, wo = float(er[-1]), float(eo[-1])         # worst per-path ratios (real)
    xmax = max(3.7, wo * 1.05)

    fig, ax = plt.subplots(figsize=(3.40, 1.62), constrained_layout=True)
    ax.axvspan(1.0, xmax, color="#000000", alpha=0.06, lw=0)
    ax.step(er, yr, where="post", color="#000000", lw=0.5,
            label=r"robust ($\kappa_{\min}{=}3$)")
    ax.step(eo, yo, where="post", color="#000000", lw=0.5, ls="--",
            label=r"optimistic ($\kappa_{\max}{=}8$)")
    ax.axvline(1.0, color="k", lw=0.5, ls="--")
    ax.text(1.03, 0.16, "funnel\nboundary", fontsize=5.6, color="0.35",
            va="center")
    ax.plot(wr, 1.0, "v", color="#000000", ms=2.5, clip_on=False)
    ax.plot(wo, 1.0, "v", color="#000000", ms=2.5, clip_on=False)
    ax.annotate(f"{wr:.2f}", (wr, 1.0), (wr, 0.85), fontsize=6,
                color="#000000", ha="center")
    ax.annotate(f"{wo:.2f}", (wo, 1.0), (wo, 0.85), fontsize=6,
                color="#000000", ha="center")
    ax.set_xlim(0.5, xmax); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\max_t \xi/\epsilon$")
    ax.set_ylabel("ECDF")
    ax.legend(loc="center right", handlelength=1.5)
    p = os.path.join(outdir, "fig_robustness.pdf")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    kappa_min, kappa_mid, kappa_max = 3.0, 5.0, 8.0
    plant_kappas = (kappa_min, kappa_mid, kappa_max)
    designs = (("rob", kappa_min), ("opt", kappa_max))

    rho0   = np.array(QUBIT_RHO0, dtype=complex)
    funnel = QUBIT_FUNNEL                              # shared geometry
    cfg = SimConfig(funnel=funnel, dt=ROBUST_DT, lam=QUBIT_LAM,
                    theta_b=QUBIT_THETA_B, **QUBIT_WEIGHTS)

    n_workers = min(10, os.cpu_count() or 8)

    print(f"robustness study (accelerated): M = {M}, "
          f"uncertainty kappa in [{kappa_min:g}, {kappa_max:g}]")
    print(f"  funnel eps0={funnel.eps0}, eps_T={funnel.eps_T}, T={funnel.T}; "
          f"theta_b={QUBIT_THETA_B}, dt={ROBUST_DT:.1e}")
    print(f"  robust controller     -> design model kappa = {kappa_min:g}")
    print(f"  optimistic controller -> design model kappa = {kappa_max:g}")
    print(f"  parallel workers: {n_workers}  "
          f"({len(plant_kappas)*len(designs)*M} trajectories)\n")

    tasks = [(kp, dk, dl, k)
             for kp in plant_kappas
             for (dl, dk) in designs
             for k in range(M)]

    t0 = time.time()
    task_fn = partial(_robust_single, cfg=cfg, rho0=rho0, seed0=SEED0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(task_fn, tasks))
    print(f"  {len(results)} trajectories integrated in {time.time()-t0:.0f}s\n")

    conf = {(kp, dl): 0 for kp in plant_kappas for dl, _ in designs}
    exc  = {(kp, dl): [] for kp in plant_kappas for dl, _ in designs}
    for kp, dl, confined, runmax in results:
        conf[(kp, dl)] += int(confined)
        exc[(kp, dl)].append(runmax)

    freq_rob, ci_rob, freq_opt, ci_opt = [], [], [], []
    exc_worst = {}
    for kp in plant_kappas:
        nr = conf[(kp, "rob")]; no = conf[(kp, "opt")]
        er = np.array(exc[(kp, "rob")]); eo = np.array(exc[(kp, "opt")])
        freq_rob.append(nr / M); ci_rob.append(clopper_pearson(nr, M))
        freq_opt.append(no / M); ci_opt.append(clopper_pearson(no, M))
        if kp == kappa_min:                           # the worst-case plant
            exc_worst["rob"], exc_worst["opt"] = er, eo
        print(f"  true plant kappa = {kp:g}:")
        print(f"    robust      confined {nr:3d}/{M} = {100*nr/M:5.1f}%   "
              f"worst xi/eps = {er.max():.2f}")
        print(f"    optimistic  confined {no:3d}/{M} = {100*no/M:5.1f}%   "
              f"worst xi/eps = {eo.max():.2f}")

    print()
    print("  On the worst-case plant the two controllers confine at "
          "comparable")
    print("  frequency; the optimistic design produces the more severe "
          "excursions")
    print("  (Proposition IV.3 bounds the worst-case drift, hence the "
          "excursion severity).")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "robustness_excursions.npz"),
             exc_rob=exc_worst["rob"], exc_opt=exc_worst["opt"],
             kappa_worst=kappa_min)
    p = _plot_robustness_cdf(outdir, exc_worst["rob"], exc_worst["opt"])
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()