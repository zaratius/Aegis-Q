"""
Backend equivalence: pure-Python reference vs Numba kernel.

The kernel (quantumdbc.kernel) is a compiled re-implementation of the
per-step mathematics of coefficients.py / controller.py / integrator.py and
the loop body of simulate.run_trajectory.  Both backends consume the SAME
numpy-drawn Wiener increments under a given seed, so their trajectories must
agree array-for-array to floating-point roundoff.  This test pins that
claim in-repo (it was previously verified only in an out-of-tree diagnostic
build).

Skips cleanly when Numba is not installed (e.g. the Windows box); run it on
any machine with Numba before trusting kernel-produced figures.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import ExpFunnel, SimConfig, run_trajectory
from quantumdbc.systems import qubit, bell
from quantumdbc import kernel as _kernel

# Tight but roundoff-tolerant: the two backends order a handful of
# floating-point reductions differently (einsum vs explicit loops).
_ATOL = 1e-10
_RTOL = 1e-10


def _compare(tag, sys_, cfg, rho0, seed=7, t_stop=None):
    tr_py = run_trajectory(sys_, cfg, rho0, seed=seed, backend="python",
                           t_stop=t_stop)
    tr_nb = run_trajectory(sys_, cfg, rho0, seed=seed, backend="numba",
                           t_stop=t_stop)
    fields = ["xi", "eps", "eps_sb", "u", "gamma", "delta", "nu",
              "betaH", "betaD", "alpha"]
    worst = 0.0
    for f in fields:
        a, b = getattr(tr_py, f), getattr(tr_nb, f)
        assert a.shape == b.shape, f"{tag}.{f}: shape {a.shape} vs {b.shape}"
        dev = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst = max(worst, dev)
        assert np.allclose(a, b, atol=_ATOL, rtol=_RTOL), (
            f"{tag}.{f}: max dev {dev:.2e} exceeds atol={_ATOL:.0e}")
    assert tr_py.case == tr_nb.case, f"{tag}: case sequences differ"
    assert tr_py.confined == tr_nb.confined
    print(f"  [ok] {tag}: python vs numba max dev {worst:.2e} "
          f"over {len(fields)} arrays, cases identical")


def test_qubit_backend_equivalence():
    if not _kernel.HAVE_NUMBA:
        print("  [skip] Numba not installed; backend test skipped")
        return
    from quantumdbc.study_config import (QUBIT_FUNNEL, QUBIT_THETA_B,
                                         QUBIT_LAM, QUBIT_RHO0, QUBIT_WEIGHTS)
    cfg = SimConfig(funnel=QUBIT_FUNNEL, lam=QUBIT_LAM, theta_b=QUBIT_THETA_B,
                    dt=5e-4, **QUBIT_WEIGHTS)
    _compare("qubit", qubit(), cfg, QUBIT_RHO0, seed=7, t_stop=1.0)


def test_bell_backend_equivalence():
    if not _kernel.HAVE_NUMBA:
        print("  [skip] Numba not installed; backend test skipped")
        return
    sys_ = bell(kappa=(50.0, 50.0, 50.0))
    e00 = np.zeros((4, 4), dtype=complex); e00[0, 0] = 1.0
    cfg = SimConfig(funnel=ExpFunnel(eps0=0.62, eps_T=0.08, T=2.0),
                    lam=0.5, theta_b=0.30, c=20.0, wgamma=1.0, wdelta=1e3,
                    umax=1.0, gmax=1.0, dt=2.5e-4, regularized=True)
    _compare("bell", sys_, cfg, e00, seed=7, t_stop=0.5)


if __name__ == "__main__":
    test_qubit_backend_equivalence()
    test_bell_backend_equivalence()
    print("backend equivalence: all checks passed")
