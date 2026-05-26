"""
Bell-state stabilization: the E2 experiment of Section V-E.

Runs the closed loop from rho_0 = |00><00| under the unregularized and the
regularized law on a shared Brownian path, prints a numerical summary, and
writes the E2 figure and the Delta t-robustness control figure.

    python scripts/run_bell.py

Findings (see STATUS.md):
  - The chattering of the unregularized law is a genuine, Delta t-robust
    obstruction (Proposition V.2), not a discretization artifact.
  - The regularized law (c = 20, w_r = 50) removes it by driving the
    coherent control to zero near the gain degeneracy beta^H_xi ~ 0; the
    terminal infidelity xi(T) is unchanged, since coherent control cannot
    reduce xi where its gain vanishes.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import bell, ExpFunnel, SimConfig, run_trajectory
from quantumdbc.figures import fig_bell_chatter, fig_chatter_robustness


def summary(tag, tr):
    u = tr.u[:, 0]
    n = len(u)
    flip = int(np.sum(np.diff(np.sign(u)) != 0))
    amp = float(np.mean(np.abs(np.diff(u))))
    print(f"  {tag}")
    print(f"    control sign-flip fraction = {flip / n:.3f}")
    print(f"    mean step-to-step |du|     = {amp:.3f}")
    print(f"    mean |u_1|                 = {np.mean(np.abs(u)):.3f}")
    print(f"    xi(0) -> xi(T)             = {tr.xi[0]:.4f} -> {tr.xi[-1]:.4f}")
    print(f"    worst negativity           = {tr.max_neg:.1e}")


def main():
    sys_ = bell()
    ket00 = np.zeros(4, dtype=complex); ket00[0] = 1.0
    rho0 = np.outer(ket00, ket00.conj())
    funnel = ExpFunnel(eps0=0.62, eps_T=0.20, T=1.0)
    dt, seed = 2.5e-4, 7

    common = dict(funnel=funnel, dt=dt, lam=0.5, s_b=0.05,
                  wgamma=1.0, wdelta=1e3)
    cfg_un = SimConfig(regularized=False, **common)
    cfg_re = SimConfig(regularized=True, wr=50.0, c=20.0, **common)

    print(f"Bell-state E2:  rho_0 = |00><00|, dt = {dt:.1e}, seed = {seed}")
    t0 = time.time()
    tr_un = run_trajectory(sys_, cfg_un, rho0, seed=seed)
    tr_re = run_trajectory(sys_, cfg_re, rho0, seed=seed)
    print(f"  two trajectories integrated in {time.time() - t0:.1f}s\n")

    summary("unregularized  (w_r = 0,  w_u = 1)", tr_un)
    summary("regularized    (w_r = 50, c = 20)", tr_re)

    print("\n  note: xi(T) is essentially equal for the two laws -- coherent")
    print("  control cannot reduce xi where beta^H_xi ~ 0, so switching it")
    print("  off (the regularized behaviour) costs nothing in performance.")

    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    print("\n  writing figures ...")
    p1 = fig_bell_chatter(outdir, seed=seed, dt=dt)
    p2 = fig_chatter_robustness(outdir, seed=seed)
    print(f"    {p1}")
    print(f"    {p2}")


if __name__ == "__main__":
    main()
