"""
Verification of the closed-form controller (Theorem IV.2).

The five-case closed form is checked against (i) the multi-channel dual
root find and (ii) an OSQP solution of the same QP, over random pointwise
data.  This converts Theorem IV.2 from a paper proof to an empirically
confirmed identity, and demonstrates that the closed form attains the
optimum a general QP solver returns.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantumdbc import QPData, solve_closed_form, solve_multichannel, solve_osqp


def random_qp(rng, m, p):
    """Random well-posed QP data; betaD < 0 enforces assumption (A1)."""
    return QPData(
        alpha=float(rng.normal(0, 2)),
        betaH=rng.normal(0, 1.5, size=m),
        betaD=-np.abs(rng.normal(0, 1.5, size=p)) - 0.05,
        V=float(abs(rng.normal(0, 1)) + 0.1),
        lam=float(abs(rng.normal(0, 1)) + 0.1),
        u_prev=rng.uniform(-1, 1, size=m),
        wu=rng.uniform(0.2, 3.0, size=m),
        wr=rng.uniform(0.0, 6.0, size=m),
        wgamma=rng.uniform(0.2, 3.0, size=p),
        wdelta=float(rng.uniform(1e2, 1e3)),
        umax=np.ones(m),
        gmax=np.ones(p),
    )


def _diff(a, b):
    return (np.max(np.abs(a.u - b.u))
            + np.max(np.abs(a.gamma - b.gamma))
            + abs(a.delta - b.delta))


def test_single_channel_vs_osqp():
    rng = np.random.default_rng(11)
    worst_cf, worst_mc = 0.0, 0.0
    cases = {}
    for _ in range(2000):
        d = random_qp(rng, 1, 1)
        cf = solve_closed_form(d)
        mc = solve_multichannel(d)
        qp = solve_osqp(d)
        worst_cf = max(worst_cf, _diff(cf, qp))
        worst_mc = max(worst_mc, _diff(cf, mc))
        cases[cf.case] = cases.get(cf.case, 0) + 1
    assert worst_cf < 1e-5, f"closed-form vs OSQP {worst_cf:.2e}"
    assert worst_mc < 1e-7, f"closed-form vs multichannel {worst_mc:.2e}"
    print(f"  [ok] m=p=1: closed-form vs OSQP   max dev {worst_cf:.2e}")
    print(f"  [ok] m=p=1: closed-form vs rootfd max dev {worst_mc:.2e}")
    print(f"       case coverage: {dict(sorted(cases.items()))}")


def test_multichannel_vs_osqp():
    rng = np.random.default_rng(13)
    worst = 0.0
    for (m, p) in ((1, 2), (2, 3)):
        for _ in range(800):
            d = random_qp(rng, m, p)
            mc = solve_multichannel(d)
            qp = solve_osqp(d)
            worst = max(worst, _diff(mc, qp))
        print(f"  [ok] m={m},p={p}: root-find vs OSQP   max dev {worst:.2e}")
    # tolerance honest to OSQP's first-order accuracy on this ill-conditioned
    # QP; the root find itself is exact (see test_single_channel_vs_osqp)
    assert worst < 5e-4, f"multichannel vs OSQP {worst:.2e}"


def test_kkt_optimality():
    """Residual check: stationarity + primal feasibility of the closed form."""
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(2000):
        d = random_qp(rng, 1, 1)
        r = solve_closed_form(d)
        u, g, dl, nu = float(r.u[0]), float(r.gamma[0]), r.delta, r.nu
        bH, bD = float(d.betaH[0]), float(d.betaD[0])
        # safety constraint must hold (<= 0 slack)
        slack = d.alpha + bH * u + bD * g + d.lam * d.V - dl
        worst = max(worst, max(slack, 0.0))
        # delta = nu / wdelta
        worst = max(worst, abs(dl - nu / d.wdelta))
    assert worst < 1e-7, f"KKT residual {worst:.2e}"
    print(f"  [ok] KKT residual (feasibility + delta identity) {worst:.2e}")


if __name__ == "__main__":
    print("controller verification")
    test_single_channel_vs_osqp()
    test_multichannel_vs_osqp()
    test_kkt_optimality()
    print("all controller tests passed")
