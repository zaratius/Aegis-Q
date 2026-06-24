"""
Accelerated E1 qubit confinement: the M = 1000 ensemble for Section V-E4.

REVISED:
  * Imports the SHARED study geometry (study_config) so this figure and the
    breach figure describe ONE system.  The old config here was
    (0.55, 0.08, s_b=0.04) -- a different funnel and buffer from the breach
    script's (0.70, 0.12, s_b=0.25), which made the feasibility-vs-confinement
    story impossible to write honestly.
  * Reports BOTH exit events (shell-exit tau_Omega, funnel-exit tau_aleph)
    and the corrected Corollary III.2 shell bound, which uses the V-space
    slack Delta(T) = sum(delta/s) dt (the 1/s factor was missing before) and
    the FINITE level V(s_b) -- not the old 2.4/Lambda -> 0 reasoning, which was
    the source of the contradiction.

    python scripts/run_e1_acc.py
"""

# -------------------------------------------------------------------------
# APPLE ACCELERATE GRIDLOCK PREVENTION (must precede any numpy import,
# including in spawned children)
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
from quantumdbc import qubit, SimConfig, run_trajectory
from quantumdbc.study_config import (
    QUBIT_FUNNEL, QUBIT_SB, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)
from quantumdbc.exit_metrics import exit_metrics
from quantumdbc.figures import fig_qubit_confinement


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def _simulate_single_path(k, sys_, cfg, rho0):
    return run_trajectory(sys_, cfg, rho0, seed=1000 + k, store=True)


def main():
    M = 1000
    sys_   = qubit()
    rho0   = np.array(QUBIT_RHO0, dtype=complex)
    funnel = QUBIT_FUNNEL                             # shared geometry
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=QUBIT_LAM, s_b=QUBIT_SB,
                    **QUBIT_WEIGHTS)

    n_workers = min(10, os.cpu_count() or 8)

    print(f"Accelerated E1 qubit confinement: M = {M}, dt = {cfg.dt:.1e}")
    print(f"  funnel eps0={funnel.eps0}, eps_T={funnel.eps_T}, T={funnel.T}; "
          f"s_b={QUBIT_SB}, lam={QUBIT_LAM}")
    print(f"  parallel workers: {n_workers}")
    t0 = time.time()

    task = partial(_simulate_single_path, sys_=sys_, cfg=cfg, rho0=rho0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        trajs = list(pool.map(task, range(M)))
    print(f"  ensemble integrated in {time.time() - t0:.0f}s")

    # --- the two exit events, computed consistently for every path ---
    mets = [exit_metrics(tr, cfg.s_b) for tr in trajs]
    n_shell  = sum(m["shell_exit"]  for m in mets)
    n_funnel = sum(m["funnel_exit"] for m in mets)
    ls, hs = clopper_pearson(n_shell, M)
    lf, hf = clopper_pearson(n_funnel, M)

    print(f"  shell-exit  (tau_Omega<=T, s<s_b, still in funnel): "
          f"{n_shell}/{M} = {100*n_shell/M:.1f}%  "
          f"95% CI [{100*ls:.1f}, {100*hs:.1f}]")
    print(f"  funnel-exit (tau_aleph<=T, xi>=eps, true breach):   "
          f"{n_funnel}/{M} = {100*n_funnel/M:.1f}%  "
          f"95% CI [{100*lf:.1f}, {100*hf:.1f}]")

    # --- corrected Corollary III.2 shell bound (finite level V(s_b)) ---
    Delta_T = float(np.mean([m["Delta_T"] for m in mets]))   # ensemble mean
    V_sb    = mets[0]["V_sb"]
    s_bar   = float(np.max(funnel.eps(trajs[0].t)))
    s0      = min(funnel.eps(trajs[0].t[0]) - float(trajs[0].xi[0]), cfg.s_b)
    V0      = float(-np.log(max(s0, 1e-12) / s_bar))
    bound   = (V0 + Delta_T) / V_sb
    print(f"  V-space Delta(T) = {Delta_T:.3f}   V(s_b) = {V_sb:.3f}   "
          f"V0 = {V0:.3f}")
    print(f"  Corollary III.2 shell bound: "
          f"P[tau_Omega <= T] <= (V0+Delta(T))/V(s_b) = {bound:.3f}")
    print("  (finite, quantitative -- replaces the old 2.4/Lambda -> 0, which "
          "predicted")
    print("   zero exits and so contradicted a nonzero measured rate)")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    p = fig_qubit_confinement(outdir, trajs, funnel.eps(trajs[0].t),
                              trajs[0].t, ci=(n_shell / M, ls, hs))
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
