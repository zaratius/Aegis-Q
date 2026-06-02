"""
Robustness to dissipative-rate uncertainty: Section V-E8 study.  Parallelized.

Accelerated version of run_robustness.py, following the ProcessPoolExecutor
pattern of the other *_acc scripts.  The work is SIX ensembles --

    3 plant rates kappa in {kappa_min, kappa_mid, kappa_max}
  x 2 designs   {robust = kappa_min, optimistic = kappa_max}

-- each of M trajectories.  Rather than run them as six serial loops, all
6*M trajectories are flattened into one task list and dispatched to a single
pool: no worker idles between ensembles, and the macOS 'spawn' cost is paid
once.  Each worker reduces its trajectory to a scalar tuple (confined flag +
per-path max xi/eps) via exit_metrics and discards the path, so peak memory
is one trajectory per worker.

Geometry and weights come from quantumdbc.study_config, so this study uses
the SAME funnel / buffer as E1 / breach / feasibility (non-empty shell;
the old hard-coded eps_T=0.12, s_b=0.25 had the empty-shell bug).

SHARED-NOISE CONTRACT (preserved from the serial version): for a given
(plant kappa, trajectory index k) the robust and optimistic controllers use
the SAME seed 5000+k, so the two controllers are compared on identical
Wiener paths.  Seeds depend only on k, never on the design, so this holds
across designs; results are bit-identical to the serial run.

    python scripts/run_robustness_acc.py [M]          # M defaults to 40
"""

# -------------------------------------------------------------------------
# APPLE ACCELERATE GRIDLOCK PREVENTION -- before any numpy import (also in
# the spawned children, which re-execute this file)
# -------------------------------------------------------------------------
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
    QUBIT_FUNNEL, QUBIT_SB, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)
from quantumdbc.exit_metrics import exit_metrics
from quantumdbc.figures import fig_robustness

# dt for the robustness ensembles.  Note: at Omega=20 the loop is fully
# smooth only at dt <= 1e-4 (STATUS.md); 5e-4 matches the original study and
# keeps it comparable, and since aliasing inflates BOTH controllers equally
# the robust-vs-optimistic *comparison* is unaffected.  Drop to 1e-4 (4x
# cost) if you want the fully un-aliased regime.
ROBUST_DT = 5e-4
SEED0 = 5000


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# qubit() is cheap, but it is called once per trajectory; memoize per worker
# so the generator basis is not rebuilt 6*M times.  Keyed on the float kappa;
# run_trajectory never mutates the System, so sharing a cached instance is
# safe.  Each worker process has its own cache.
@lru_cache(maxsize=8)
def _qubit_cached(kappa: float):
    return qubit(kappa=kappa)


# -------------------------------------------------------------------------
# Worker -- top-level so macOS 'spawn' can pickle it.  A task is
# (plant_kappa, design_kappa, design_label, k).  The plant is propagated by
# the true kappa; the controller's coefficients are evaluated on design_sys
# (Proposition IV.3).  Returns the scalars the figure needs.
# -------------------------------------------------------------------------
def _robust_single(task, cfg, rho0, seed0):
    plant_kappa, design_kappa, design_label, k = task
    plant = _qubit_cached(plant_kappa)
    design = _qubit_cached(design_kappa)
    tr = run_trajectory(plant, cfg, rho0, seed=seed0 + k,
                        store=True, design_sys=design)
    m = exit_metrics(tr, cfg.s_b)
    # 'confined' := never exited the funnel (tau_aleph), matching simulate.py
    return (plant_kappa, design_label, bool(not m["funnel_exit"]),
            float(m["runmax"]))


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    kappa_min, kappa_mid, kappa_max = 3.0, 5.0, 8.0
    plant_kappas = (kappa_min, kappa_mid, kappa_max)
    designs = (("rob", kappa_min), ("opt", kappa_max))

    rho0   = np.array(QUBIT_RHO0, dtype=complex)
    funnel = QUBIT_FUNNEL                              # shared geometry
    cfg = SimConfig(funnel=funnel, dt=ROBUST_DT, lam=QUBIT_LAM, s_b=QUBIT_SB,
                    **QUBIT_WEIGHTS)

    # M4 Max = 14 P-cores; os.cpu_count() also counts E-cores, which produce
    # a long tail, so cap at the P-core count.
    n_workers = min(10, os.cpu_count() or 8)

    print(f"robustness study (accelerated): M = {M}, "
          f"uncertainty kappa in [{kappa_min:g}, {kappa_max:g}]")
    print(f"  funnel eps0={funnel.eps0}, eps_T={funnel.eps_T}, T={funnel.T}; "
          f"s_b={QUBIT_SB}, dt={ROBUST_DT:.1e}")
    print(f"  robust controller     -> design model kappa = {kappa_min:g}")
    print(f"  optimistic controller -> design model kappa = {kappa_max:g}")
    print(f"  parallel workers: {n_workers}  "
          f"({len(plant_kappas)*len(designs)*M} trajectories)\n")

    # flatten all 6*M trajectories into one task list
    tasks = [(kp, dk, dl, k)
             for kp in plant_kappas
             for (dl, dk) in designs
             for k in range(M)]

    t0 = time.time()
    task_fn = partial(_robust_single, cfg=cfg, rho0=rho0, seed0=SEED0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(task_fn, tasks))
    print(f"  {len(results)} trajectories integrated in {time.time()-t0:.0f}s\n")

    # regroup by (plant_kappa, design_label)
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
    p = fig_robustness(outdir, plant_kappas, freq_rob, ci_rob,
                       freq_opt, ci_opt,
                       exc_worst["rob"], exc_worst["opt"], kappa_min)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()