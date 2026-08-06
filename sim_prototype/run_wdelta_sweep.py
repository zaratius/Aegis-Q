"""
w_delta sweep: shows that the persistent shell- and funnel-exit rates of
Section V-E4 are controlled by w_delta, not by the integration step Delta t.

Runs three w_delta values at a single fine Delta t, prints the two exit
rates for each, and confirms that they scale ~ 1/w_delta. This is the
empirical evidence that the persistent rates are the QP's finite-w_delta
floor (interior slack delta* = nu*/w_delta), not infeasibility or
integration slack.

    python scripts/run_wdelta_sweep.py [M]              # M defaults to 500
"""
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, time, concurrent.futures
from functools import partial

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, SimConfig, run_trajectory
from quantumdbc.study_config import (
    QUBIT_FUNNEL, QUBIT_THETA_B, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)
from quantumdbc.exit_metrics import exit_metrics


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def _one(k, sys_, cfg, rho0, seed_base):
    tr = run_trajectory(sys_, cfg, rho0, seed=seed_base + k, store=True)
    m = exit_metrics(tr, cfg.theta_b)
    return (m["shell_exit"], m["funnel_exit"])


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    dt = 1e-4                                          # fixed, un-aliased
    wdeltas = (1e3, 1e4, 1e5)

    sys_   = qubit()
    rho0   = np.array(QUBIT_RHO0, dtype=complex)
    funnel = QUBIT_FUNNEL
    n_workers = min(10, os.cpu_count() or 8)

    print(f"w_delta sweep: M={M}, dt={dt:.0e}, "
          f"funnel eps_T={funnel.eps_T}, theta_b={QUBIT_THETA_B}")
    print(f"  parallel workers: {n_workers}\n")

    weights = dict(QUBIT_WEIGHTS)        # copy so we can override wdelta
    shell_rates, funnel_rates = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for j, wd in enumerate(wdeltas):
            weights["wdelta"] = wd
            cfg = SimConfig(funnel=funnel, dt=dt, lam=QUBIT_LAM,
                            theta_b=QUBIT_THETA_B, **weights)
            seed_base = 9000 + 1000 * j
            task = partial(_one, sys_=sys_, cfg=cfg, rho0=rho0,
                           seed_base=seed_base)
            t0 = time.time()
            res = list(pool.map(task, range(M)))
            n_sh = sum(r[0] for r in res)
            n_fu = sum(r[1] for r in res)
            ls, hs = clopper_pearson(n_sh, M)
            lf, hf = clopper_pearson(n_fu, M)
            shell_rates.append(n_sh / M)
            funnel_rates.append(n_fu / M)
            print(f"  w_delta={wd:.0e}:  "
                  f"shell {n_sh:3d}/{M}={100*n_sh/M:5.1f}% "
                  f"[{100*ls:4.1f},{100*hs:4.1f}]   "
                  f"funnel {n_fu:3d}/{M}={100*n_fu/M:5.1f}% "
                  f"[{100*lf:4.1f},{100*hf:4.1f}]   "
                  f"({time.time()-t0:.0f}s)")

    print("\n  scaling check (rate ratios across 10x w_delta steps):")
    for i in range(len(wdeltas) - 1):
        sr = shell_rates[i] / max(shell_rates[i+1], 1e-6)
        fr = funnel_rates[i] / max(funnel_rates[i+1], 1e-6)
        print(f"    w_delta {wdeltas[i]:.0e} -> {wdeltas[i+1]:.0e}:  "
              f"shell ratio {sr:.2f}x   funnel ratio {fr:.2f}x   "
              f"(expect ~10x if rates ~ 1/w_delta)")


if __name__ == "__main__":
    main()
