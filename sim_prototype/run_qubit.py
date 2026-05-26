"""
Demonstration of the closed-loop qubit controller (Algorithm 1).

Runs one verified, numerically stable closed-loop trajectory and prints a
summary.  The parameters here are the *corrected* values discussed in
STATUS.md -- they differ from the Section V-E table draft (Delta t = 1e-4
rather than 1e-3; a feasible funnel) for the reasons documented there.

    python scripts/run_qubit.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import qubit, ExpFunnel, SimConfig, run_trajectory


def main():
    sys_ = qubit()                       # Gamma_m=1, Omega=20, kappa=5, eta=0.6
    rho0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)

    funnel = ExpFunnel(eps0=0.55, eps_T=0.06, T=6.0)
    cfg = SimConfig(funnel=funnel, dt=1e-4, lam=0.6, s_b=0.04,
                    wr=5.0, wgamma=1.0, wdelta=1e3, regularized=True)

    t0 = time.time()
    tr = run_trajectory(sys_, cfg, rho0, seed=1)
    dt_wall = time.time() - t0

    flips = int(np.sum(np.diff(np.sign(tr.u[:, 0])) != 0))
    print(f"closed-loop qubit run: {len(tr.t)} steps, {dt_wall:.1f}s")
    print(f"  xi(0)            = {tr.xi[0]:.4f}")
    print(f"  xi(T)            = {tr.xi[-1]:.4f}")
    print(f"  confined         = {tr.confined}")
    print(f"  max xi/epsilon   = {np.max(tr.xi / tr.eps):.3f}")
    print(f"  control sign flips = {flips}  (smooth if O(1))")
    print(f"  worst negativity = {tr.max_neg:.1e}")
    print(f"  case histogram   = "
          f"{ {c: tr.case.count(c) for c in sorted(set(tr.case))} }")


if __name__ == "__main__":
    main()
