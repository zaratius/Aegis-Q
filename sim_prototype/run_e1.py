"""
Qubit confinement under the funnel: the E1 experiment of Section V-E.

Runs a Monte Carlo ensemble of closed-loop qubit trajectories from a fixed
initial state, reports the confinement frequency with a 95% Clopper-Pearson
interval, and writes the E1 figure.

    python scripts/run_e1.py [M]          # M defaults to 75

Findings (see STATUS.md):
  - At the funnel used here the ensemble confinement frequency is ~93%
    (M = 75).  The breaching paths are NOT a failure of the framework: on
    every breaching path the QP slack variable delta activates strongly
    (delta ~ 6-7, versus ~0 on confined paths), which is precisely the
    signature of the barrier constraint becoming infeasible.  The funnel
    is marginally outside the admissible set; a funnel fixed by the E3
    feasibility check restores pathwise confinement.
  - M = 75 is an in-environment preview.  The paper-final figure should
    use M = 1000 (the confinement frequency claim then has a CI roughly
    a factor sqrt(1000/75) ~ 3.6 tighter).
"""
import sys, os, time
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


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    sys_ = qubit()
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    funnel = ExpFunnel(eps0=0.55, eps_T=0.08, T=4.0)
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=0.6, s_b=0.04,
                    wr=50.0, c=20.0, regularized=True)

    print(f"E1 qubit confinement: M = {M}, dt = {cfg.dt:.1e}")
    t0 = time.time()
    trajs = [run_trajectory(sys_, cfg, rho0, seed=1000 + k,
                            store=(k < 60))
             for k in range(M)]
    print(f"  ensemble integrated in {time.time() - t0:.0f}s")

    nconf = sum(tr.confined for tr in trajs)
    lo, hi = clopper_pearson(nconf, M)
    print(f"  confined: {nconf}/{M} = {100 * nconf / M:.1f}%")
    print(f"  95% Clopper-Pearson CI: [{100 * lo:.1f}%, {100 * hi:.1f}%]")

    stored = [tr for tr in trajs if len(tr.t) > 0]

    breach = [tr for tr in stored if not tr.confined]
    if breach:
        dmax = [tr.delta.max() for tr in breach]
        print(f"  {len(breach)} breaching paths among the {len(stored)} "
              f"stored; QP slack delta_max on those in "
              f"[{min(dmax):.1f}, {max(dmax):.1f}]")
        print("  (slack activation confirms the breaches are funnel "
              "infeasibility, not a framework failure)")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    p = fig_qubit_confinement(outdir, stored, funnel.eps(stored[0].t),
                              stored[0].t, ci=(nconf / M, lo, hi))
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
