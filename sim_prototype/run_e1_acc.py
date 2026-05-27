import os
import sys
import time

# 1. APPLE ACCELERATE GRIDLOCK PREVENTION
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy import stats
import concurrent.futures
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, ExpFunnel, SimConfig, run_trajectory
from quantumdbc.figures import fig_qubit_confinement

def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

# 2. ISOLATED WORKER FOR MACOS 'SPAWN'
def _simulate_single_path(k, sys_, cfg, rho0):
    # FIX: Store all paths. 1000 paths ~ 2GB RAM. 
    # Guarantees the pointwise frequency curve calculates across all M=1000.
    return run_trajectory(sys_, cfg, rho0, seed=1000 + k, store=True)

def main():
    M = 1000
    sys_ = qubit()
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    funnel = ExpFunnel(eps0=0.55, eps_T=0.08, T=4.0)
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=0.6, s_b=0.04,
                    wr=50.0, c=20.0, regularized=True)

    # 3. P-CORE OPTIMIZATION
    # Set to your Ultra's specific Performance Core count (e.g., 16 or 24).
    # This prevents slower E-cores from causing a "long tail" wait time at the end.
    PERFORMANCE_CORES = 24 

    print(f"Accelerated E1 qubit confinement: M = {M}, dt = {cfg.dt:.1e}")
    t0 = time.time()

    task = partial(_simulate_single_path, sys_=sys_, cfg=cfg, rho0=rho0)
    
    # 4. PICKLING DRY RUN
    # Tests 2 paths to ensure the dataclasses serialize correctly on macOS spawn.
    print("  running pickling dry-run...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        _ = list(executor.map(task, range(2)))
    print("  dry-run successful. Launching full ensemble...")

    # 5. FULL EXECUTION
    with concurrent.futures.ProcessPoolExecutor(max_workers=PERFORMANCE_CORES) as executor:
        trajs = list(executor.map(task, range(M)))

    print(f"  ensemble integrated in {time.time() - t0:.0f}s")

    # Because store=True for all, trajs and stored are identical.
    # The figure will correctly compute the curve over M=1000.
    nconf = sum(tr.confined for tr in trajs)
    lo, hi = clopper_pearson(nconf, M)
    
    print(f"  confined: {nconf}/{M} = {100 * nconf / M:.1f}%")
    print(f"  95% Clopper-Pearson CI: [{100 * lo:.1f}%, {100 * hi:.1f}%]")

    breach = [tr for tr in trajs if not tr.confined]
    if breach:
        dmax = [tr.delta.max() for tr in breach]
        print(f"  {len(breach)} breaching paths detected; "
              f"QP slack delta_max on those in "
              f"[{min(dmax):.1f}, {max(dmax):.1f}]")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    
    # Pass the full ensemble to guarantee math integrity in the figure
    p = fig_qubit_confinement(outdir, trajs, funnel.eps(trajs[0].t),
                              trajs[0].t, ci=(nconf / M, lo, hi))
    print(f"  wrote {p}")

if __name__ == "__main__":
    main()