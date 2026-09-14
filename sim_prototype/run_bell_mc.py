"""
Certified Bell Monte Carlo: the widened-funnel ensemble study.

Modes (results accumulate in bell_mc_results.json):

    python run_bell_mc.py [M]           main ensemble (default M = 1000)
                                        + fig_bell_confinement.pdf
    python run_bell_mc.py refine [M]    dt in {2.5e-4, 1.25e-4, 6.25e-5},
                                        default M = 300 per point
    python run_bell_mc.py envelope      certified-bound refinement ladder
    python run_bell_mc.py summary       print everything accumulated
"""

# -------------------------------------------------------------------------
# BLAS THREAD PINNING (must precede any numpy import, including in spawned
# children; harmless on Windows/Linux, required on Apple Accelerate)
# -------------------------------------------------------------------------
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# The BLAS pins above would also strangle numba's OpenMP threading layer;
# give the envelope kernel (prange over the t grid, pure scalar math, no
# BLAS inside) the full machine explicitly. Must precede any numba import.
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 8))

import sys
import json
import time
import platform
import concurrent.futures
from functools import partial

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import (SimConfig, run_trajectory,
                        bell_slack_envelope, closed_loop_bound)
from quantumdbc.study_config import (BELL_FUNNEL, BELL_THETA_B, BELL_LAM,
                                     BELL_DT, BELL_WEIGHTS, BELL_THETA_PREP,
                                     bell_system, bell_rho0_prep)
from quantumdbc.exit_metrics import exit_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "bell_mc_results.json")
SEED0 = 3000


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% confidence interval for a binomial proportion."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def _load():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def _save(data):
    data["_meta"] = dict(
        geometry=dict(eps0=BELL_FUNNEL.eps0, eps_T=BELL_FUNNEL.eps_T,
                      T=BELL_FUNNEL.T, theta_b=BELL_THETA_B, lam=BELL_LAM,
                      dt=BELL_DT, weights={k: v for k, v in
                                           BELL_WEIGHTS.items()}),
        platform=platform.platform(), numpy=np.__version__,
    )
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, RESULTS)


def _one_path(k, cfg=None, seed0=SEED0):
    """Worker: integrate one path, return slim per-path results.

    The system and initial state are rebuilt in-process (cheap) so only the
    SimConfig crosses the process boundary; the returned xi trace is float32
    to keep M = 1000 transfers light.
    """
    sys_ = bell_system()
    rho0 = bell_rho0_prep()          # imperfect preparation
    tr = run_trajectory(sys_, cfg, rho0, seed=seed0 + k, store=True)
    met = exit_metrics(tr, cfg.theta_b)
    return dict(metrics=met, xi=tr.xi.astype(np.float32),
                confined=bool(tr.confined),
                u1=tr.u[:, 0].astype(np.float32),
                bH1=np.abs(tr.betaH[:, 0]).astype(np.float32))


def _run_ensemble(cfg, M, seed0=SEED0, workers=None):
    n_workers = workers or min(10, os.cpu_count() or 8)
    task = partial(_one_path, cfg=cfg, seed0=seed0)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        out = list(pool.map(task, range(M), chunksize=max(1, M // (4 * n_workers))))
    return out, time.time() - t0, n_workers


def _rates(out, M):
    mets = [o["metrics"] for o in out]
    n_shell = sum(m["shell_exit"] for m in mets)
    n_funnel = sum(m["funnel_exit"] for m in mets)
    ls, hs = clopper_pearson(n_shell, M)
    lf, hf = clopper_pearson(n_funnel, M)
    Delta_T = float(np.mean([m["Delta_T"] for m in mets]))
    return dict(M=M,
                shell_exits=int(n_shell), shell_rate=n_shell / M,
                shell_ci=[ls, hs],
                funnel_exits=int(n_funnel), funnel_rate=n_funnel / M,
                funnel_ci=[lf, hf],
                Delta_T_realized=Delta_T)


def cmd_main(M=1000):
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    sys_ = bell_system()
    xi0 = 1.0 - float(np.trace(sys_.Pi @ bell_rho0_prep()).real)
    V0 = float(-np.log(1.0 - xi0 / BELL_FUNNEL.eps0))
    level = float(np.log(1.0 / BELL_THETA_B))

    print(f"Certified Bell MC: M = {M}, dt = {cfg.dt:.1e}")
    print(f"  funnel eps0={BELL_FUNNEL.eps0}, eps_T={BELL_FUNNEL.eps_T}, "
          f"T={BELL_FUNNEL.T}; theta_b={BELL_THETA_B}, lam={BELL_LAM}")
    print(f"  rho0: (Rx({BELL_THETA_PREP}) x I)|00>  -- imperfect preparation")
    print(f"  xi0 = {xi0:.3f} < shell edge "
          f"{(1 - BELL_THETA_B) * BELL_FUNNEL.eps0:.3f}  "
          f"(V0 = {V0:.4f}, Doob floor = {V0 / level:.4f})")

    out, wall, n_workers = _run_ensemble(cfg, M)
    print(f"  ensemble integrated in {wall:.0f}s on {n_workers} workers")

    r = _rates(out, M)
    print(f"  shell-exit  (tau_Omega<=T): {r['shell_exits']}/{M} = "
          f"{100 * r['shell_rate']:.1f}%  95% CI "
          f"[{100 * r['shell_ci'][0]:.1f}, {100 * r['shell_ci'][1]:.1f}]")
    print(f"  funnel-exit (tau_aleph<=T): {r['funnel_exits']}/{M} = "
          f"{100 * r['funnel_rate']:.1f}%  95% CI "
          f"[{100 * r['funnel_ci'][0]:.1f}, {100 * r['funnel_ci'][1]:.1f}]")

    bound_real = (V0 + r["Delta_T_realized"]) / level
    print(f"  realized slack Delta(T) = {r['Delta_T_realized']:.4f}  ->  "
          f"a-posteriori bound {bound_real:.3f}")

    data = _load()
    cert = data.get("envelope", {}).get("bound")
    if cert is not None:
        ok = r["shell_ci"][1] <= cert
        print(f"  certified a-priori bound {cert:.4f}: observed CI upper "
              f"{100 * r['shell_ci'][1]:.1f}% -> "
              f"{'VERIFIED' if ok else 'VIOLATED'}")
    else:
        print("  (no envelope in results file yet -- run "
              "'python run_bell_mc.py envelope' for the certified bound)")

    # figure
    from quantumdbc.figures import fig_bell_confinement
    t = np.arange(1, int(round(BELL_FUNNEL.T / cfg.dt)) + 1) * cfg.dt
    eps_curve = BELL_FUNNEL.eps(t)
    sb_curve = (1.0 - BELL_THETA_B) * eps_curve
    outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(outdir, exist_ok=True)
    n_conf = M - r["funnel_exits"]
    lc, hc = clopper_pearson(n_conf, M)
    p = fig_bell_confinement(
        outdir, t, eps_curve, sb_curve,
        [o["xi"] for o in out],
        [not o["confined"] for o in out],
        ci=(n_conf / M, lc, hc), bound=cert,
        u_paths=[o["u1"] for o in out],
        bH_paths=[o["bH1"] for o in out])
    print(f"  wrote {p}")

    r.update(dict(dt=cfg.dt, seed0=SEED0, wall_s=wall, V0=V0, level=level,
                  bound_realized=bound_real))
    data["main"] = r
    _save(data)
    print(f"  results -> {RESULTS}")


def cmd_refine(M=300, dts=(2.5e-4, 1.25e-4, 6.25e-5)):
    data = _load()
    ref = data.get("refine", {})
    for dt in dts:
        cfg = SimConfig(funnel=BELL_FUNNEL, dt=dt, lam=BELL_LAM,
                        theta_b=BELL_THETA_B, **BELL_WEIGHTS)
        print(f"refinement point dt = {dt:.2e}, M = {M}")
        out, wall, _ = _run_ensemble(cfg, M, seed0=SEED0 + 100_000)
        r = _rates(out, M)
        r.update(dict(dt=dt, wall_s=wall))
        print(f"  shell {100 * r['shell_rate']:.1f}% "
              f"[{100 * r['shell_ci'][0]:.1f}, {100 * r['shell_ci'][1]:.1f}]"
              f"   funnel {100 * r['funnel_rate']:.1f}% "
              f"[{100 * r['funnel_ci'][0]:.1f}, {100 * r['funnel_ci'][1]:.1f}]"
              f"   ({wall:.0f}s)")
        ref[f"{dt:.2e}"] = r
        data["refine"] = ref
        _save(data)
    print(f"  results -> {RESULTS}")


def cmd_envelope():
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    sys_ = bell_system()
    xi0 = 1.0 - float(np.trace(sys_.Pi @ bell_rho0_prep()).real)
    ladder = []
    for (n_t, n_xi, n_p1, n_p2) in ((81, 81, 41, 21),
                                    (161, 161, 81, 41),
                                    (241, 321, 161, 81)):
        t_grid = np.linspace(0.0, BELL_FUNNEL.T, n_t)
        t0 = time.time()
        tg, dbar, diag = bell_slack_envelope(sys_, cfg, t_grid=t_grid,
                                             n_xi=n_xi, n_p1=n_p1,
                                             n_p2=n_p2)
        bound, parts = closed_loop_bound(cfg, tg, dbar, xi0)
        wall = time.time() - t0
        row = dict(grid=[n_t, n_xi, n_p1, n_p2],
                   Delta=parts["Delta"], V0=parts["V0"],
                   level=parts["level"], bound=bound,
                   dbar_max=float(np.max(dbar)),
                   screen_max=diag["screen_max"],
                   n_solves=diag["n_solves"], wall_s=wall)
        ladder.append(row)
        print(f"  grid {n_t}x{n_xi}x{n_p1}x{n_p2}: Delta = {parts['Delta']:.6f}"
              f"  bound = {bound:.6f}  (max dbar {row['dbar_max']:.2e}, "
              f"{wall:.0f}s)")
    data = _load()
    data["envelope"] = dict(ladder=ladder, bound=ladder[-1]["bound"],
                            Delta=ladder[-1]["Delta"],
                            V0=ladder[-1]["V0"], level=ladder[-1]["level"])
    _save(data)
    print(f"  certified bound (finest grid): {ladder[-1]['bound']:.6f}")
    print(f"  results -> {RESULTS}")


def cmd_summary():
    data = _load()
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_main()
    elif args[0] == "refine":
        cmd_refine(int(args[1]) if len(args) > 1 else 300)
    elif args[0] == "envelope":
        cmd_envelope()
    elif args[0] == "summary":
        cmd_summary()
    else:
        cmd_main(int(args[0]))
