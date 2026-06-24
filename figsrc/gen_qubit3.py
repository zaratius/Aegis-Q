import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, time
import concurrent.futures
from quantumdbc.systems import qubit
from quantumdbc.barrier import ExpFunnel
from quantumdbc.simulate import SimConfig, run_trajectory

# Fig 4 trajectory bundle. M is the ensemble size (60 = dense visual; raise for
# a denser plot or to match the M=1000 rate run). Parallel across P-cores.
M = 60
NSUB = 500                       # plotted points per trajectory (subsample)
RHO0 = np.array([[0.78, 0.08], [0.08, 0.22]], dtype=complex)


def _funnel():
    return ExpFunnel(eps0=0.70, eps_T=0.30, T=4.0)


def _cfg():
    return SimConfig(funnel=_funnel(), dt=5e-4, lam=0.5, s_b=0.25, wr=50.0,
                     c=20.0, wgamma=1.0, wdelta=1e3, regularized=True)


def _worker(seed):
    """One trajectory -> (subsampled xi, confined). Module-level: spawn-safe."""
    tr = run_trajectory(qubit(), _cfg(), RHO0, seed=int(seed), store=True)
    idx = np.linspace(0, len(tr.xi) - 1, NSUB).astype(int)
    return tr.xi[idx], bool(tr.confined)


if __name__ == "__main__":
    t0 = time.time()
    cfg, fun = _cfg(), _funnel()
    n = int(round(fun.T / cfg.dt))
    idx = np.linspace(0, n - 1, NSUB).astype(int)
    t_sub = (np.arange(n) * cfg.dt)[idx]
    n_workers = min(10, os.cpu_count() or 8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_worker, [5000 + k for k in range(M)]))
    xis = np.array([r[0] for r in results])
    breach = sum(1 for r in results if not r[1])
    eps = fun.eps(t_sub)
    np.savez('qubit_data.npz', t=t_sub, eps=eps, xis=xis, s_b=cfg.s_b)
    print(f"DONE {time.time()-t0:.0f}s, M={M}, funnel-breaches={breach} "
          f"({100*breach/M:.1f}%), workers={n_workers}")
