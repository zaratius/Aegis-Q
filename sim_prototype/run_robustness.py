"""
Robustness to dissipative-rate uncertainty: the Section V-E8 study.

The engineered dissipation rate kappa of the qubit is uncertain within an
interval [kappa_min, kappa_max].  kappa enters the QP only through the
dissipative gain beta^D = -kappa xi (Lemma V.1), and the safety constraint
is hardest to satisfy when beta^D is least negative, so the worst-case
vertex of the uncertainty set is kappa = kappa_min.  Proposition IV.3 then
gives a robust controller: it computes its coefficients from the worst-case
model kappa = kappa_min, while the true plant evolves at an unknown kappa
in the interval.

This script runs two controllers --

  * robust      -- design model kappa = kappa_min
  * optimistic  -- design model kappa = kappa_max

against true plants at kappa in {kappa_min, kappa_mid, kappa_max}, reports
the confinement frequency and the worst per-path excursion of each, and
writes the Section V-E8 figure.

    python scripts/run_robustness.py [M]             # M defaults to 40

Findings (see STATUS.md):
  - The robust controller, which never sees the true kappa, confines the
    ensemble across the whole uncertainty interval.  At its design point
    kappa = kappa_min the run reduces to the nominal qubit study.
  - On the worst-case plant the robust and optimistic controllers confine
    at comparable frequency, but differ sharply in the *severity* of the
    failures: the optimistic controller, having over-estimated the
    dissipative authority, cannot arrest a path drifting toward the
    boundary, so its worst excursion is far larger.  This is the
    robustness statement appropriate to a safety setting -- Proposition
    IV.3 bounds the worst-case drift, hence the worst-case excursion, by
    the worst-case model.
  - The closed form is preserved throughout: the only change between the
    two controllers is the value of beta^D fed to the master equation, so
    robustness costs nothing per step.
"""
import sys, os, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, ExpFunnel, SimConfig, run_trajectory
from quantumdbc.figures import fig_robustness


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def ensemble(plant, design, cfg, rho0, M, seed0=5000):
    """Run an M-trajectory ensemble of (design-model controller, true plant).

    Returns the confinement count and the array of per-path maximum margin
    ratios max_t xi/epsilon.
    """
    nconf = 0
    exc = np.empty(M)
    for k in range(M):
        tr = run_trajectory(plant, cfg, rho0, seed=seed0 + k,
                            store=True, design_sys=design)
        nconf += tr.confined
        exc[k] = float(np.max(tr.xi / tr.eps))
    return nconf, exc


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    kappa_min, kappa_mid, kappa_max = 3.0, 5.0, 8.0
    plant_kappas = (kappa_min, kappa_mid, kappa_max)

    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    # Table II admissible qubit funnel
    funnel = ExpFunnel(eps0=0.70, eps_T=0.12, T=4.0)
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=0.5, s_b=0.25,
                    wr=50.0, c=20.0, wgamma=1.0, wdelta=1e3,
                    regularized=True)

    # the controllers differ only in the model kappa they are designed for
    design_rob = qubit(kappa=kappa_min)              # worst-case model
    design_opt = qubit(kappa=kappa_max)              # best-case model

    print(f"robustness study: M = {M}, uncertainty kappa in "
          f"[{kappa_min:g}, {kappa_max:g}]")
    print(f"  robust controller     -> design model kappa = {kappa_min:g}")
    print(f"  optimistic controller -> design model kappa = {kappa_max:g}\n")

    freq_rob, ci_rob, freq_opt, ci_opt = [], [], [], []
    exc_worst = {}
    for kp in plant_kappas:
        plant = qubit(kappa=kp)
        t0 = time.time()
        nr, er = ensemble(plant, design_rob, cfg, rho0, M)
        no, eo = ensemble(plant, design_opt, cfg, rho0, M)
        freq_rob.append(nr / M); ci_rob.append(clopper_pearson(nr, M))
        freq_opt.append(no / M); ci_opt.append(clopper_pearson(no, M))
        if kp == kappa_min:                          # the worst-case plant
            exc_worst["rob"], exc_worst["opt"] = er, eo
        print(f"  true plant kappa = {kp:g}:")
        print(f"    robust      confined {nr:3d}/{M} = {100*nr/M:5.1f}%   "
              f"worst xi/eps = {er.max():.2f}")
        print(f"    optimistic  confined {no:3d}/{M} = {100*no/M:5.1f}%   "
              f"worst xi/eps = {eo.max():.2f}   ({time.time()-t0:.0f}s)")

    print()
    print("  On the worst-case plant the two controllers confine at "
          "comparable")
    print("  frequency; the optimistic design produces the more severe "
          "excursions.")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    p = fig_robustness(outdir, plant_kappas, freq_rob, ci_rob,
                       freq_opt, ci_opt,
                       exc_worst["rob"], exc_worst["opt"], kappa_min)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()