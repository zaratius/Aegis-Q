"""
E3 -- funnel feasibility check (Section V-E).

For the qubit closed loop, evaluates the feasibility condition

    |eps_dot(t)|  <=  V(t),
    V(t) = sum |beta^H_i| u_max + sum |beta^D_j| gamma_max
           - ( mu + (1/2) kappa_V sigma^2 ),

on the funnel boundary shell xi > shell * eps -- the region where the
barrier is active and the condition is meaningful.  The figure overlays the
E1 breach onsets: if E3 is correct, every E1 breach falls inside an E3
infeasible window.

    python scripts/run_e3.py [M]

M defaults to 75 (about 4 min at dt = 5e-4); the same seeds as run_e1.py
are used so the breach onsets correspond exactly.

Findings (see STATUS.md):
  - On the boundary shell the drift/noise pressure mu + (1/2) kappa_V
    sigma^2 (median ~7) exceeds the bounded control authority sum |beta|
    u_max (median ~2.4) by a factor of about three.  The funnel is
    infeasible near the boundary not because it contracts too fast
    (|eps_dot| ~ 0.3 is small) but because the measurement backaction
    overwhelms the actuators there.
  - The pressure scales as 1/s_b through the barrier curvature kappa_V =
    1/s.  Raising the buffer s_b is the corrective lever: at s_b ~ 0.25 the
    pressure drops below the control authority and the funnel becomes
    feasible.  This is the change the Section V-E table needs, not a
    gentler eps(t).
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, ExpFunnel, SimConfig, run_trajectory
from quantumdbc.figures import fig_feasibility, feasibility_data


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    shell = 0.85
    sys_ = qubit()
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)
    funnel = ExpFunnel(eps0=0.55, eps_T=0.08, T=4.0)
    cfg = SimConfig(funnel=funnel, dt=5e-4, lam=0.6, s_b=0.04,
                    wr=50.0, c=20.0, wgamma=1.0, wdelta=1e3,
                    regularized=True)

    print(f"E3 funnel feasibility: M = {M}, boundary shell xi > {shell} eps")
    t0 = time.time()
    trajs = [run_trajectory(sys_, cfg, rho0, seed=1000 + k)
             for k in range(min(M, 60))]
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
