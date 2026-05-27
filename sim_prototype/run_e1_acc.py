"""
Accelerated E1 qubit confinement: the M = 1000 ensemble for Section V-E4.

Parallelised version of run_e1.py.  Each closed-loop trajectory is run in
its own process; the per-step BLAS threads are pinned to one each so the
processes don't fight over Apple Accelerate's thread pool.

    python scripts/run_e1_acc.py
"""

# -------------------------------------------------------------------------
# APPLE ACCELERATE GRIDLOCK PREVENTION
# These MUST be set before NumPy is imported anywhere in the process,
# including in the child workers (the macOS 'spawn' start method
# re-executes this file in each child, so the order matters here too).
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
from quantumdbc.figures import fig_qubit_confinement


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# -------------------------------------------------------------------------
# Worker: top-level so macOS 'spawn' can pickle it.  store=True for all
# paths: at dt = 5e-4, T = 4 each trajectory occupies ~1.2 MB, so the full
# M = 1000 ensemble resident in the parent is ~1.2 GB -- comfortable on a
# Mac Studio and required for the figure's pointwise confinement-frequency
# curve to average over all M, not just a subset.
# -------------------------------------------------------------------------
def _simulate_single_path(k, sys_, cfg, rho0):
    return run_trajectory(sys_, cfg, rho0, seed=1000 + k, store=True)


def main():
    M = 1000
    sys_ = qubit()
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    funnel = ExpFunnel(eps0=0.55, eps_T=0.08, T=4.0)
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=0.6, s_b=0.04,
                    wr=50.0, c=20.0, regularized=True)

    # Cap at the P-core count of the target machine.  os.cpu_count() on
    # Apple Silicon counts P + E cores; assigning workers to E-cores
    # produces a long tail of stragglers at the end of the run.
    # M2 Ultra = 16 P-cores, M3 Max = 12, M2 Max = 8.  Adjust to taste.
    n_workers = min(16, os.cpu_count() or 8)

    print(f"Accelerated E1 qubit confinement: M = {M}, dt = {cfg.dt:.1e}")
    print(f"  parallel workers: {n_workers}")
    t0 = time.time()

    task = partial(_simulate_single_path, sys_=sys_, cfg=cfg, rho0=rho0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        trajs = list(pool.map(task, range(M)))

    print(f"  ensemble integrated in {time.time() - t0:.0f}s")

    # Confinement statistics over the full M = 1000 ensemble.
    nconf = sum(tr.confined for tr in trajs)
    lo, hi = clopper_pearson(nconf, M)
    print(f"  confined: {nconf}/{M} = {100 * nconf / M:.1f}%")
    print(f"  95% Clopper-Pearson CI: [{100 * lo:.1f}%, {100 * hi:.1f}%]")

    # Slack-on-breach diagnostic: store=True for every path guarantees
    # tr.delta is populated, so this no longer needs to filter on len(tr.t).
    breach = [tr for tr in trajs if not tr.confined]
    if breach:
        dmax = [tr.delta.max() for tr in breach]
        print(f"  {len(breach)} breaching paths; QP slack delta_max in "
              f"[{min(dmax):.1f}, {max(dmax):.1f}]")
        print("  (slack activation on breaching paths is the empirical "
              "signature of the degraded guarantee, Corollary III.2)")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    # Pass the full ensemble: the pointwise confinement-frequency curve is
    # then a clean M = 1000 estimate, matching the headline CI in the title.
    p = fig_qubit_confinement(outdir, trajs, funnel.eps(trajs[0].t),
                              trajs[0].t, ci=(nconf / M, lo, hi))
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()