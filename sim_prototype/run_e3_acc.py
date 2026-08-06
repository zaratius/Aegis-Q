"""
E3 -- funnel feasibility check (Section V-E).  Parallelized.

For the qubit closed loop, evaluates the funnel-gauge feasibility condition

    |eps_dot(t)|  <=  V(t),
    V(t) = (eps/xi) [ sum |beta^H_i| u_max + sum |beta^D_j| gamma_max
                      - mu - (1/2) kappa_V sigma^2 ],

on the funnel boundary shell xi > shell * eps -- the region where the
barrier is active and the condition is meaningful.  The figure overlays the
E1 breach onsets: if E3 is correct, every E1 breach falls inside an E3
infeasible window.

    python scripts/run_e3.py [M]

The seeds are seed = 1000 + k, IDENTICAL to run_e1.py / run_e1_acc.py, so
the breach onsets correspond to exactly the same sample paths.  The feasibility
analysis reads the per-step coefficient arrays (betaH, betaD, alpha, xi), so
every path must be stored; the ensemble is therefore capped at 60 paths to
bound memory, exactly as the serial version.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, SimConfig, run_trajectory
from quantumdbc.figures import fig_feasibility, feasibility_data
from quantumdbc.study_config import (
    QUBIT_FUNNEL, QUBIT_THETA_B, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)


# -------------------------------------------------------------------------
# Worker -- top-level for macOS 'spawn'.  store=True is mandatory here: the
# feasibility analysis consumes the per-step coefficient arrays, so paths
# cannot be run with store=False as in the breach study.  Seed = 1000 + k
# matches run_e1 exactly.
# -------------------------------------------------------------------------
def _simulate_single_path(k, sys_, cfg, rho0):
    return run_trajectory(sys_, cfg, rho0, seed=1000 + k, store=True)


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    # n_paths = min(M, 60)                  # stored paths -- memory bound
    n_paths = M
    shell = 0.85
    sys_ = qubit()
    rho0 = np.array(QUBIT_RHO0, dtype=complex)

    funnel = QUBIT_FUNNEL                              # shared geometry
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=QUBIT_LAM,
                    theta_b=QUBIT_THETA_B, **QUBIT_WEIGHTS)

    xi0 = 1.0 - float(QUBIT_RHO0[0, 0].real)
    # relative buffer: the shell {xi <= (1-theta_b) eps} never empties, so the
    # old empty-shell guard is gone; only the Problem 1 precondition remains
    # (already checked as a hard guard in study_config.py; re-asserted here
    # for a fast, local failure if this script is ever run standalone).
    assert xi0 < (1.0 - cfg.theta_b) * funnel.eps(0), \
        "CRITICAL: Problem 1 precondition violated. Funnel too narrow at t=0."

    # M4 Max = 14 P-cores; os.cpu_count() also counts the 2 E-cores, which
    # produce a long tail, so cap at the P-core count.
    n_workers = 10

    print(f"E3 funnel feasibility: M = {n_paths}, boundary shell xi > {shell} eps")
    print(f"  parallel workers: {n_workers}")
    t0 = time.time()

    task = partial(_simulate_single_path, sys_=sys_, cfg=cfg, rho0=rho0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        trajs = list(pool.map(task, range(n_paths)))
    print(f"  ensemble integrated in {time.time() - t0:.0f}s")

    t, demand, V_min, V_med, infeas = feasibility_data(
        trajs, funnel, cfg.umax, cfg.gmax, shell)
    valid = ~np.isnan(V_min)
    n_infeas = int(infeas[valid].sum())
    print(f"  boundary-shell time points: {int(valid.sum())}")
    print(f"  infeasible (V_min < |eps_dot|): {n_infeas} "
          f"({100 * n_infeas / max(valid.sum(), 1):.1f}%)")

    breach = [tr.t[np.where(tr.xi >= tr.eps)[0][0]]
              for tr in trajs if np.any(tr.xi >= tr.eps)]
    print(f"  E1 breaches in this ensemble: {len(breach)}")
    if breach:
        in_window = sum(infeas[np.argmin(np.abs(t - bt))] for bt in breach)
        print(f"  breaches falling inside an E3 infeasible window: "
              f"{in_window}/{len(breach)}")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    p = fig_feasibility(outdir, trajs, funnel, cfg.umax, cfg.gmax, shell)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
