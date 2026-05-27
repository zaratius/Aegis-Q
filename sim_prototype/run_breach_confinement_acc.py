"""
Breach-rate step refinement: Figure 1 of Section V-E.  Parallelized.

Re-runs the admissible-funnel qubit ensemble at a sequence of integration
steps Delta t and reports the funnel-breach rate at each, with a 95%
Clopper-Pearson interval.  For an admissible funnel the breach rate should
fall along an O(Delta t) line and vanish in the continuum limit, confirming
that the finite-step boundary excursions of the E1 study are integration
slack (Corollary III.2) rather than funnel infeasibility.

    python scripts/run_breach_confinement.py [M]      # M defaults to 400

A single ProcessPoolExecutor is held open across all Delta t values, so the
process-spawn cost is paid once.  store=False everywhere -- the study only
needs the scalar 'confined' flag per trajectory, never the per-step arrays.

Three Delta t are run.  Wall-clock dominated by the finest step.
"""

# -------------------------------------------------------------------------
# APPLE ACCELERATE GRIDLOCK PREVENTION -- before any numpy import
# -------------------------------------------------------------------------
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import concurrent.futures
from functools import partial

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, ExpFunnel, SimConfig, run_trajectory
from quantumdbc.figures import fig_breach_refinement


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# -------------------------------------------------------------------------
# Worker -- top-level so macOS 'spawn' can pickle it.  Returns the scalar
# 'confined' flag only; the per-step arrays are not needed here and are not
# materialised (store=False).  Seed offset depends on the dt index so the
# three ensembles use disjoint, reproducible seed ranges.
# -------------------------------------------------------------------------
def _breach_single(k, sys_, cfg, rho0, seed_base):
    tr = run_trajectory(sys_, cfg, rho0, seed=seed_base + k, store=False)
    return not tr.confined                # True iff this path breached


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    dts = (1e-3, 5e-4, 2.5e-4)

    sys_ = qubit()                                    # kappa = 5 nominal
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    # Table II admissible qubit funnel -- large buffer s_b = 0.25
    funnel = ExpFunnel(eps0=0.70, eps_T=0.12, T=4.0)

    # Cap workers at the P-core count of the target machine; os.cpu_count()
    # on Apple Silicon counts P + E cores, and E-cores produce a long tail.
    # Adjust to your hardware: M2 Ultra = 16, M3 Max = 12, M2 Max = 8.
    n_workers = min(16, os.cpu_count() or 8)

    print(f"breach-rate step refinement: M = {M}, "
          f"dts = {', '.join(f'{d:.1e}' for d in dts)}")
    print(f"  parallel workers: {n_workers}")

    rates, los, his = [], [], []

    # One pool for the whole run: spawn cost is paid once, not per dt.
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for j, dt in enumerate(dts):
            cfg = SimConfig(funnel=funnel, dt=dt, lam=0.5, s_b=0.25,
                            wr=50.0, c=20.0, wgamma=1.0, wdelta=1e3,
                            regularized=True)
            # disjoint seed ranges per dt: seeds 2000..2000+M-1 for dt[0],
            # 3000..3000+M-1 for dt[1], etc.  Reproducible across reruns.
            seed_base = 2000 + 1000 * j
            task = partial(_breach_single, sys_=sys_, cfg=cfg, rho0=rho0,
                           seed_base=seed_base)

            t0 = time.time()
            breached = list(pool.map(task, range(M)))
            nbreach = sum(breached)
            lo, hi = clopper_pearson(nbreach, M)
            rates.append(nbreach / M)
            los.append(lo)
            his.append(hi)
            print(f"  dt = {dt:.1e}: breach {nbreach:4d}/{M} = "
                  f"{100*nbreach/M:5.1f}%  95% CI "
                  f"[{100*lo:4.1f}, {100*hi:4.1f}]   "
                  f"({time.time()-t0:.0f}s)")

    # Diagnostic line: report the direction.  The paper's story requires
    # rates to be monotonically decreasing; flag if they are not.
    if all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1)):
        print("  -> rates decrease with dt: consistent with O(dt) slack")
    elif all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)):
        print("  -> rates INCREASE with dt: NOT consistent with O(dt) slack;"
              " breaches may be a continuous-time phenomenon")
    else:
        print("  -> rates non-monotone; insufficient resolution or noise")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    p = fig_breach_refinement(outdir, dts, rates, los, his)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()