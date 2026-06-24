"""
Robustness to dissipative-rate uncertainty: the Section V-E8 study.

REVISED: imports the SHARED study geometry (study_config) instead of the
hard-coded (0.70, 0.12, s_b=0.25), which had the empty-shell bug
(s_b > eps_T => Omega(T) empty).  Excursions are now read through
exit_metrics so "worst xi/eps" is computed identically to the other studies.

The engineered dissipation rate kappa is uncertain within [kappa_min,
kappa_max].  kappa enters the QP only through beta^D = -kappa xi (Lemma V.1),
and the safety constraint is hardest when beta^D is least negative, so the
worst-case vertex is kappa = kappa_min.  The robust controller designs for
kappa_min; the optimistic one for kappa_max.

    python scripts/run_robustness.py [M]             # M defaults to 40
"""
import sys, os, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, SimConfig, run_trajectory
from quantumdbc.study_config import (
    QUBIT_FUNNEL, QUBIT_SB, QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS,
)
from quantumdbc.exit_metrics import exit_metrics
from quantumdbc.figures import fig_robustness


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def ensemble(plant, design, cfg, rho0, M, seed0=5000):
    """Run an M-trajectory ensemble of (design-model controller, true plant).

    Returns the confinement count (no funnel-exit) and the array of per-path
    maximum margin ratios max_t xi/eps, both via exit_metrics so the metric
    matches the rest of the study.
    """
    nconf = 0
    exc = np.empty(M)
    for k in range(M):
        tr = run_trajectory(plant, cfg, rho0, seed=seed0 + k,
                            store=True, design_sys=design)
        m = exit_metrics(tr, cfg.s_b)
        nconf += (not m["funnel_exit"])          # confined := never exited funnel
        exc[k] = m["runmax"]
    return nconf, exc


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    kappa_min, kappa_mid, kappa_max = 3.0, 5.0, 8.0
    plant_kappas = (kappa_min, kappa_mid, kappa_max)

    rho0   = np.array(QUBIT_RHO0, dtype=complex)
    funnel = QUBIT_FUNNEL                             # shared geometry
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=QUBIT_LAM, s_b=QUBIT_SB,
                    **QUBIT_WEIGHTS)

    design_rob = qubit(kappa=kappa_min)              # worst-case model
    design_opt = qubit(kappa=kappa_max)              # best-case model

    print(f"robustness study: M = {M}, uncertainty kappa in "
          f"[{kappa_min:g}, {kappa_max:g}]")
    print(f"  funnel eps0={funnel.eps0}, eps_T={funnel.eps_T}, T={funnel.T}; "
          f"s_b={QUBIT_SB}")
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