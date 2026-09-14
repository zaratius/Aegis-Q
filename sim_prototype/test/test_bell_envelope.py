"""Validation of the Bell certificate machinery (bell_slack_envelope).

The envelope rests on four load-bearing claims; each is tested here against
the generic trace-formula coefficients on random states:

  1. FEATURE EXACTNESS: with H0 = 0, every QP coefficient at a Bell state
     rho is an exact function of (xi, P1, P2, P3):
         mu == 0,   betaD_k = -kappa_k P_k  (kappa_k = Tr(Lc_k^dag Lc_k)),
         sigma = -4 sqrt(eta Gm) (1 - xi)(P2 + P3),   sum_k P_k = xi.

  2. BETA_H COLLAPSE: the dual root nu* is maximized at betaH = 0 (each
     coherent term of the residual is pointwise non-increasing in |betaH|,
     including through the state weight w_u = c/(|betaH|+eps_f)), and
     betaH = 0 is attained by Bell-diagonal states.

  3. BISECTION EQUIVALENCE: the vectorized bisection inside the envelope
     agrees with solve_multichannel on the same QP data.

  4. DOMINATION: the gridded envelope dominates the true objective
     (delta* - lam V)/s evaluated at sampled shell states -- including
     adversarial Bell-diagonal states planted in the small-xi boundary
     layer where the objective is positive.

Plus: python == numba backend parity for the certified BELL config.

Run:  python test/test_bell_envelope.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from quantumdbc import (SimConfig, QPData, solve_multichannel,
                        bell_slack_envelope, run_trajectory)
from quantumdbc.coefficients import coefficients_generic, infidelity
from quantumdbc.study_config import (BELL_FUNNEL, BELL_THETA_B, BELL_LAM,
                                     BELL_DT, BELL_WEIGHTS, BELL_KAPPA,
                                     bell_system, bell_rho0, bell_rho0_prep)

RNG = np.random.default_rng(20260808)


def _bell_basis(sys_):
    """Columns: |Phi+>, |Phi->, |Psi+>, |Psi-> (computational-basis order
    |00>, |01>, |10>, |11>)."""
    s = 1 / np.sqrt(2)
    phi_p = np.array([s, 0, 0, s], dtype=complex)
    phi_m = np.array([s, 0, 0, -s], dtype=complex)
    psi_p = np.array([0, s, s, 0], dtype=complex)
    psi_m = np.array([0, s, -s, 0], dtype=complex)
    return np.column_stack([phi_p, phi_m, psi_p, psi_m])


def _random_states(sys_, n, xi_max):
    """Random density matrices with xi <= xi_max: Ginibre mixed toward Pi,
    plus random pure states mixed toward Pi, plus Bell-diagonal states."""
    out = []
    Pi = sys_.Pi
    for _ in range(n):
        kind = RNG.integers(0, 3)
        if kind == 0:                                    # Ginibre
            G = RNG.normal(size=(4, 4)) + 1j * RNG.normal(size=(4, 4))
            rho = G @ G.conj().T
            rho /= np.trace(rho).real
        elif kind == 1:                                  # random pure
            v = RNG.normal(size=4) + 1j * RNG.normal(size=4)
            v /= np.linalg.norm(v)
            rho = np.outer(v, v.conj())
        else:                                            # Bell-diagonal
            p = RNG.dirichlet(np.ones(4))
            B = _bell_basis(sys_)
            rho = (B * p) @ B.conj().T
        xi = infidelity(sys_, rho)
        if xi > 1e-12:
            w = min(1.0, RNG.uniform(0.0, xi_max) / xi)
            rho = (1.0 - w) * Pi + w * rho
        out.append(rho)
    return out


def _features(sys_, rho):
    """(xi, P1, P2, P3) in the {Phi-, Psi+, Psi-} channel order of sys.Lc."""
    xi = infidelity(sys_, rho)
    B = _bell_basis(sys_)
    kets = {"Phi+": B[:, 0], "Phi-": B[:, 1],
            "Psi+": B[:, 2], "Psi-": B[:, 3]}
    P = [float((kets[n].conj() @ rho @ kets[n]).real)
         for n in ("Phi-", "Psi+", "Psi-")]
    return xi, P


def test_feature_exactness(n=2000):
    sys_ = bell_system()
    kap = [float(np.trace(Lc.conj().T @ Lc).real) for Lc in sys_.Lc]
    assert np.allclose(kap, BELL_KAPPA), f"kappa extraction: {kap}"
    Gm = float(np.abs(sys_.L[0, 0]) ** 2)
    worst = dict(mu=0.0, betaD=0.0, sigma=0.0, sumP=0.0)
    for rho in _random_states(sys_, n, xi_max=0.9):
        c = coefficients_generic(sys_, rho)
        xi, P = _features(sys_, rho)
        worst["mu"] = max(worst["mu"], abs(c["mu"]))
        worst["sumP"] = max(worst["sumP"], abs(sum(P) - xi))
        for j in range(3):
            worst["betaD"] = max(worst["betaD"],
                                 abs(c["betaD"][j] + kap[j] * P[j]))
        sig_pred = -4.0 * np.sqrt(sys_.eta * Gm) * (1.0 - xi) * (P[1] + P[2])
        worst["sigma"] = max(worst["sigma"], abs(c["sigma"] - sig_pred))
    for k, v in worst.items():
        assert v < 1e-10, f"feature identity '{k}' violated: {v:.2e}"
    print(f"  [ok] feature exactness on {n} random states "
          f"(worst dev {max(worst.values()):.2e})")


def test_betaH_zero_on_diagonal(n=200):
    sys_ = bell_system()
    B = _bell_basis(sys_)
    worst = 0.0
    for _ in range(n):
        p = RNG.dirichlet(np.ones(4))
        rho = (B * p) @ B.conj().T
        c = coefficients_generic(sys_, rho)
        worst = max(worst, float(np.max(np.abs(c["betaH"]))))
    assert worst < 1e-12, f"betaH on Bell-diagonal states: {worst:.2e}"
    print(f"  [ok] betaH == 0 on {n} Bell-diagonal states "
          f"(worst {worst:.2e})")


def test_betaH_monotonicity(n=4000):
    """nu*(betaH = 0) >= nu*(betaH != 0) at identical remaining data."""
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    n_viol = 0
    worst = 0.0
    for _ in range(n):
        alpha = RNG.uniform(0.0, 3.0)
        V = RNG.uniform(0.0, np.log(1.0 / cfg.theta_b))
        xi = RNG.uniform(1e-5, 0.68)
        P = RNG.dirichlet(np.ones(3)) * xi
        betaD = -BELL_KAPPA * P
        bH = RNG.uniform(-40.0, 40.0, size=2)
        wu_r = cfg.c / (np.abs(bH) + cfg.eps_f)
        common = dict(alpha=alpha, betaD=betaD, V=V, lam=cfg.lam,
                      wgamma=np.full(3, cfg.wgamma), wdelta=cfg.wdelta,
                      umax=np.full(2, cfg.umax), gmax=np.full(3, cfg.gmax))
        r0 = solve_multichannel(QPData(betaH=np.zeros(2),
                                       wu=np.full(2, cfg.c / cfg.eps_f),
                                       **common))
        r1 = solve_multichannel(QPData(betaH=bH, wu=wu_r, **common))
        gap = r1.nu - r0.nu
        if gap > 1e-8:
            n_viol += 1
            worst = max(worst, gap)
    assert n_viol == 0, f"{n_viol}/{n} monotonicity violations, worst {worst:.2e}"
    print(f"  [ok] nu*(betaH=0) >= nu*(betaH) on {n} random QP instances")


def test_bisection_vs_multichannel(n=2000):
    """The envelope's closed-form bisection == solve_multichannel at betaH=0."""
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    kap = np.full(3, BELL_KAPPA)
    worst = 0.0
    for _ in range(n):
        alpha = RNG.uniform(0.0, 3.0)
        V = RNG.uniform(0.0, np.log(1.0 / cfg.theta_b))
        xi = 10.0 ** RNG.uniform(-6, np.log10(0.68))
        P = RNG.dirichlet(np.ones(3)) * xi
        d = QPData(alpha=alpha, betaH=np.zeros(2), betaD=-kap * P,
                   V=V, lam=cfg.lam,
                   wu=np.full(2, cfg.c / cfg.eps_f),
                   wgamma=np.full(3, cfg.wgamma), wdelta=cfg.wdelta,
                   umax=np.full(2, cfg.umax), gmax=np.full(3, cfg.gmax))
        ref = solve_multichannel(d, tol=1e-14)

        def g(nu):
            gam = np.minimum(nu * kap * P / cfg.wgamma, cfg.gmax)
            return alpha + cfg.lam * V - nu / cfg.wdelta - float(kap * P @ gam)

        lo, hi = 0.0, cfg.wdelta * (alpha + cfg.lam * V) + 1.0
        if g(0.0) <= 0.0:
            nu = 0.0
        else:
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if g(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            nu = 0.5 * (lo + hi)
        worst = max(worst, abs(nu - ref.nu) / max(1.0, abs(ref.nu)))
    assert worst < 1e-8, f"bisection vs multichannel rel dev {worst:.2e}"
    print(f"  [ok] vectorized-bisection residual == solve_multichannel "
          f"on {n} instances (worst rel dev {worst:.2e})")


def test_envelope_dominates(n_random=3000, n_planted=400):
    """dbar(t) >= true objective at sampled shell states (incl. planted
    boundary-layer adversaries), up to grid resolution in t."""
    sys_ = bell_system()
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    t_grid = np.linspace(0.0, BELL_FUNNEL.T, 81)
    tg, dbar, _ = bell_slack_envelope(sys_, cfg, t_grid=t_grid,
                                      n_xi=161, n_p1=81, n_p2=41)

    B = _bell_basis(sys_)

    def true_q(rho, t):
        eps_t = float(BELL_FUNNEL.eps(t))
        xi = infidelity(sys_, rho)
        if xi > (1.0 - cfg.theta_b) * eps_t:
            return None                                   # outside the shell
        s = max(eps_t - xi, cfg.theta_b * eps_t)
        V = -np.log(s / eps_t)
        c = coefficients_generic(sys_, rho)
        alpha = (c["mu"] - (xi / eps_t) * float(BELL_FUNNEL.eps_dot(t))
                 + 0.5 * c["sigma"] ** 2 / s)
        wu = (cfg.c / (np.abs(c["betaH"]) + cfg.eps_f) if cfg.regularized
              else np.ones(2))
        d = QPData(alpha=alpha, betaH=c["betaH"], betaD=c["betaD"],
                   V=V, lam=cfg.lam, wu=wu,
                   wgamma=np.full(3, cfg.wgamma), wdelta=cfg.wdelta,
                   umax=np.full(2, cfg.umax), gmax=np.full(3, cfg.gmax))
        res = solve_multichannel(d)
        return (res.delta - cfg.lam * V) / s

    n_checked, worst_gap = 0, -np.inf
    # random states at random grid times
    for _ in range(n_random):
        t = float(RNG.choice(tg))
        eps_t = float(BELL_FUNNEL.eps(t))
        rho = _random_states(sys_, 1, (1.0 - cfg.theta_b) * eps_t)[0]
        q = true_q(rho, t)
        if q is None:
            continue
        k = int(np.argmin(np.abs(tg - t)))
        gap = q - dbar[k]
        worst_gap = max(worst_gap, gap)
        assert gap <= 1e-9, f"domination violated at t={t}: q={q}, dbar={dbar[k]}"
        n_checked += 1
    # planted boundary-layer adversaries: Bell-diagonal, xi ~ the sliver
    # where the objective is positive
    n_pos = 0
    for _ in range(n_planted):
        t = float(RNG.choice(tg))
        xi = 10.0 ** RNG.uniform(-4.5, -2.0)
        frac = RNG.dirichlet(np.ones(3))
        p = np.concatenate([[1.0 - xi], frac * xi])
        rho = (B * p) @ B.conj().T
        q = true_q(rho, t)
        if q is None:
            continue
        if q > 0.0:
            n_pos += 1
        k = int(np.argmin(np.abs(tg - t)))
        gap = q - dbar[k]
        worst_gap = max(worst_gap, gap)
        assert gap <= 1e-9, (f"domination violated at planted state "
                             f"t={t}, xi={xi:.1e}: q={q}, dbar={dbar[k]}")
        n_checked += 1
    assert n_pos > 0, "no planted adversary had positive objective -- " \
                      "the boundary layer test has no teeth"
    print(f"  [ok] envelope dominates the true objective at {n_checked} "
          f"sampled shell states ({n_pos} with positive objective; "
          f"worst gap {worst_gap:.2e})")


def test_envelope_backend_parity():
    """The numba envelope kernel reproduces the NumPy reference path."""
    import time
    sys_ = bell_system()
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    t_grid = np.linspace(0.0, BELL_FUNNEL.T, 41)
    t0 = time.time()
    _, d_py, g_py = bell_slack_envelope(sys_, cfg, t_grid=t_grid,
                                        n_xi=81, n_p1=41, n_p2=21,
                                        backend="python")
    t_py = time.time() - t0
    bell_slack_envelope(sys_, cfg, t_grid=t_grid[:2], n_xi=81, n_p1=41,
                        n_p2=21, backend="numba")        # warm the JIT
    t0 = time.time()
    _, d_nb, g_nb = bell_slack_envelope(sys_, cfg, t_grid=t_grid,
                                        n_xi=81, n_p1=41, n_p2=21,
                                        backend="numba")
    t_nb = time.time() - t0
    assert np.allclose(d_nb, d_py, rtol=1e-6, atol=1e-12), \
        f"envelope backend mismatch: max dev {np.max(np.abs(d_nb - d_py)):.2e}"
    assert g_nb["n_solves"] == g_py["n_solves"], \
        f"solve counts differ: {g_nb['n_solves']} vs {g_py['n_solves']}"
    assert abs(g_nb["screen_max"] - g_py["screen_max"]) < 1e-9
    print(f"  [ok] envelope numba == python "
          f"(max dev {np.max(np.abs(d_nb - d_py)):.2e}; "
          f"{t_py:.1f}s -> {t_nb:.2f}s, x{t_py/max(t_nb,1e-9):.0f})")


def test_backend_parity_bell_config():
    sys_ = bell_system()
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    rho0 = bell_rho0_prep()          # the state the certified MC integrates
    tr_py = run_trajectory(sys_, cfg, rho0, seed=42, store=True,
                           backend="python", t_stop=0.05)
    tr_nb = run_trajectory(sys_, cfg, rho0, seed=42, store=True,
                           backend="numba", t_stop=0.05)
    dev = max(float(np.max(np.abs(tr_py.xi - tr_nb.xi))),
              float(np.max(np.abs(tr_py.delta - tr_nb.delta))),
              float(np.max(np.abs(tr_py.gamma - tr_nb.gamma))))
    assert dev < 1e-9, f"python vs numba dev {dev:.2e} on BELL config"
    print(f"  [ok] python == numba on the certified BELL config, prepared "
          f"rho0 (max dev {dev:.2e} over 200 steps)")


def test_determinism_from_exact_00():
    """Documented structural fact: from exactly |00><00| the loop is
    noise-decoupled (parity measurement degenerate on the even sector,
    u == 0), so distinct seeds give the identical path."""
    sys_ = bell_system()
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    rho0 = bell_rho0()
    c = coefficients_generic(sys_, rho0)
    assert abs(c["sigma"]) < 1e-14 and np.max(np.abs(c["betaH"])) < 1e-14
    t1 = run_trajectory(sys_, cfg, rho0, seed=1, store=True, t_stop=0.1)
    t2 = run_trajectory(sys_, cfg, rho0, seed=2**31 - 1, store=True,
                        t_stop=0.1)
    dev = float(np.max(np.abs(t1.xi - t2.xi)))
    assert dev < 1e-12, f"paths from exact |00> differ across seeds: {dev:.2e}"
    assert float(np.max(np.abs(t1.u))) == 0.0
    print(f"  [ok] exact |00> start is seed-independent (max dev {dev:.2e}); "
          f"the certified MC must start from bell_rho0_prep()")


def test_envelope_rejects_nonzero_H0():
    from quantumdbc.systems import bell
    cfg = SimConfig(funnel=BELL_FUNNEL, dt=BELL_DT, lam=BELL_LAM,
                    theta_b=BELL_THETA_B, **BELL_WEIGHTS)
    try:
        bell_slack_envelope(bell(), cfg)                  # default omega = 5
    except ValueError as e:
        assert "H0" in str(e)
        print("  [ok] envelope rejects H0 != 0")
        return
    raise AssertionError("envelope accepted H0 != 0")


if __name__ == "__main__":
    print("Bell certificate validation:")
    test_feature_exactness()
    test_betaH_zero_on_diagonal()
    test_betaH_monotonicity()
    test_bisection_vs_multichannel()
    test_envelope_dominates()
    test_envelope_backend_parity()
    test_backend_parity_bell_config()
    test_determinism_from_exact_00()
    test_envelope_rejects_nonzero_H0()
    print("bell envelope validation: all checks passed")
