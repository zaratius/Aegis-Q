"""
Accelerated E1 qubit confinement: the M = 1000 ensemble for Section V-E4.

    python scripts/run_e1_acc.py
"""

# APPLE ACCELERATE GRIDLOCK PREVENTION
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
    QUBIT_FUNNEL, QUBIT_THETA_B, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)
from quantumdbc.exit_metrics import exit_metrics
from quantumdbc.figures import fig_qubit_confinement


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

SEED0 = 5000

def _simulate_single_path(k, sys_, cfg, rho0):
    return run_trajectory(sys_, cfg, rho0, seed=SEED0 + k, store=True)


def main():
    M = 1000
    sys_   = qubit()
    rho0   = np.array(QUBIT_RHO0, dtype=complex)
    funnel = QUBIT_FUNNEL                             # shared geometry
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=QUBIT_LAM,
                    theta_b=QUBIT_THETA_B, **QUBIT_WEIGHTS)

    n_workers = min(10, os.cpu_count() or 8)

    print(f"Accelerated E1 qubit confinement: M = {M}, dt = {cfg.dt:.1e}")
    print(f"  funnel eps0={funnel.eps0}, eps_T={funnel.eps_T}, T={funnel.T}; "
          f"theta_b={QUBIT_THETA_B}, lam={QUBIT_LAM}")
    print(f"  parallel workers: {n_workers}")
    t0 = time.time()

    task = partial(_simulate_single_path, sys_=sys_, cfg=cfg, rho0=rho0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        trajs = list(pool.map(task, range(M)))
    print(f"  ensemble integrated in {time.time() - t0:.0f}s")

    # the two exit events, computed consistently for every path
    mets = [exit_metrics(tr, cfg.theta_b) for tr in trajs]
    n_shell  = sum(m["shell_exit"]  for m in mets)
    n_funnel = sum(m["funnel_exit"] for m in mets)
    ls, hs = clopper_pearson(n_shell, M)
    lf, hf = clopper_pearson(n_funnel, M)

    print(f"  shell-exit  (tau_Omega<=T, s<theta_b*eps, still in funnel): "
          f"{n_shell}/{M} = {100*n_shell/M:.1f}%  "
          f"95% CI [{100*ls:.1f}, {100*hs:.1f}]")
    print(f"  funnel-exit (tau_aleph<=T, xi>=eps, true breach):   "
          f"{n_funnel}/{M} = {100*n_funnel/M:.1f}%  "
          f"95% CI [{100*lf:.1f}, {100*hf:.1f}]")

    # funnel-gauge Corollary III.2 shell bound (constant exit level)
    Delta_T = float(np.mean([m["Delta_T"] for m in mets]))   # ensemble mean
    V_level = mets[0]["V_level"]                             # log(1/theta_b)
    xi0     = float(trajs[0].xi[0])
    V0      = float(-np.log(1.0 - xi0 / funnel.eps0))
    bound   = (V0 + Delta_T) / V_level
    print(f"  V-space Delta(T) = {Delta_T:.3f}   "
          f"level log(1/theta_b) = {V_level:.3f}   V0 = {V0:.3f}")
    print(f"  Corollary III.2 shell bound (realized slack): "
          f"P[tau_Omega <= T] <= (V0+Delta(T))/log(1/theta_b) = {bound:.3f}")
    print("  (realized-slack version; the a-priori envelope bound of the")
    print("   closed-loop corollary comes from quantumdbc.envelope --")
    print("   qubit_nu_envelope + closed_loop_bound -- and is what Table I")
    print("   and Section V-E4 report)")

    from quantumdbc import qubit_slack_envelope, closed_loop_bound
    t0 = time.time()
    tg, dbar = qubit_slack_envelope(sys_, cfg, n_x3=161, n_x2=81)
    cert, _ = closed_loop_bound(cfg, tg, dbar, 1.0 - float(QUBIT_RHO0[0, 0].real))
    print(f"  certified a-priori bound (envelope, coarse grid): "
          f"P[tau_Omega <= T] <= {cert:.4f}  ({time.time()-t0:.0f}s)")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    n_conf = M - n_funnel                  
    lc, hc = clopper_pearson(n_conf, M)
    p = fig_qubit_confinement(outdir, trajs, funnel.eps(trajs[0].t),
                              trajs[0].t, ci=(n_conf / M, lc, hc),
                              sb_curve=(1.0 - QUBIT_THETA_B)
                              * funnel.eps(trajs[0].t),
                              bound=cert)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()