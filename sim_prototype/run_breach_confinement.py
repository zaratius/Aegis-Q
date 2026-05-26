"""
Breach-rate step refinement: Figure 1 of Section V-E.

Re-runs the admissible-funnel qubit ensemble at a sequence of integration
steps Delta t and reports the funnel-breach rate at each, with a 95%
Clopper-Pearson interval.  For an admissible funnel the breach rate falls
along an O(Delta t) line and vanishes in the continuum limit, which
confirms that the finite-step boundary excursions of the E1 study are
integration slack (Corollary III.2), not funnel infeasibility.

    python scripts/run_breach_refinement.py [M]      # M defaults to 40

Findings (see STATUS.md):
  - With the admissible (large-buffer) qubit funnel of Table II the breach
    rate decreases monotonically as Delta t is halved and reaches zero at
    the finest step, tracking the O(Delta t) reference.  The boundary
    excursions are therefore discretization slack, not infeasibility: the
    funnel lies in the admissible set E.
  - This is the companion of run_e3.py.  E3 shows the *inadmissible* funnel
    is breached by a correct controller; this study shows the *admissible*
    funnel is breached only by the integrator, and only at finite step.
"""
import sys, os, time
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


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    dts = (1e-3, 5e-4, 2.5e-4)

    sys_ = qubit()                                    # kappa = 5 nominal
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    # Table II admissible qubit funnel -- large buffer s_b = 0.25
    funnel = ExpFunnel(eps0=0.70, eps_T=0.12, T=4.0)

    print(f"breach-rate step refinement: M = {M}, "
          f"dts = {', '.join(f'{d:.1e}' for d in dts)}")
    rates, los, his = [], [], []
    for dt in dts:
        cfg = SimConfig(funnel=funnel, dt=dt, lam=0.5, s_b=0.25,
                        wr=50.0, c=20.0, wgamma=1.0, wdelta=1e3,
                        regularized=True)
        t0 = time.time()
        nbreach = 0
        for k in range(M):
            # store=False: only the 'confined' flag is needed
            tr = run_trajectory(sys_, cfg, rho0, seed=2000 + k, store=False)
            nbreach += (not tr.confined)
        lo, hi = clopper_pearson(nbreach, M)
        rates.append(nbreach / M); los.append(lo); his.append(hi)
        print(f"  dt = {dt:.1e}: breach {nbreach}/{M} = {100*nbreach/M:5.1f}%  "
              f"95% CI [{100*lo:.1f}, {100*hi:.1f}]   "
              f"({time.time()-t0:.0f}s)")

    if rates[0] > 0 and rates[-1] < rates[0]:
        print("  -> breach rate decreasing with Delta t (integration slack)")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    p = fig_breach_refinement(outdir, dts, rates, los, his)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()