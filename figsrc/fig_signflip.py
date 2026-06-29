"""Fig 7 -- unregularized control sign-flip fraction vs integration step, qubit
and Bell. Both rise to a plateau as dt->0: a continuous-time effect, not a
discretization artifact (which would vanish). Reproduces fig_chatter_robustness."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
from quantumdbc import qubit, bell, ExpFunnel, SimConfig, run_trajectory

SEED = 7; dts = (2e-3, 1e-3, 5e-4, 2.5e-4)
sq = qubit(); rq = np.array([[0.78, 0.08], [0.08, 0.22]], complex)
fq = ExpFunnel(eps0=0.55, eps_T=0.10, T=1.0)
sb = bell(); ket = np.zeros(4, complex); ket[0] = 1.0; rb = np.outer(ket, ket.conj())
fb = ExpFunnel(eps0=0.62, eps_T=0.20, T=1.0)
ffq, ffb = [], []
for dt in dts:
    tq = run_trajectory(sq, SimConfig(funnel=fq, dt=dt, lam=0.6, s_b=0.04, regularized=False), rq, seed=SEED)
    ffq.append(np.mean(np.diff(np.sign(tq.u[:, 0])) != 0))
    tb = run_trajectory(sb, SimConfig(funnel=fb, dt=dt, lam=0.5, s_b=0.05, regularized=False), rb, seed=SEED)
    ffb.append(np.mean(np.diff(np.sign(tb.u[:, 0])) != 0))
dns = np.asarray(dts) * 1e3
fig, ax = plt.subplots(figsize=(ps.COL, 1.60), constrained_layout=True)
ax.semilogx(dns, ffq, 'o-', color="#000000", lw=0.5, ms=2.5, label='qubit')
ax.semilogx(dns, ffb, 's-', color="#000000", lw=0.5, ms=2.5, label=r'Bell $|00\rangle$')
ax.set_xlabel(r'integration step $\Delta t$ (ns)'); ax.set_ylabel('sign-flip fraction')
ax.set_ylim(0, 1); ax.invert_xaxis()
ax.legend(loc='lower left', handlelength=1.3)
ax.annotate(r'plateau as $\Delta t\!\to\!0$', xy=(dns[-1], ffb[-1] - 0.14),
            xytext=(dns[1], 0.28), fontsize=6, color='0.45',
            arrowprops=dict(arrowstyle='->', color='0.5', lw=0.5))
fig.savefig('../figures/fig_chatter_robustness.pdf'); print("wrote fig_chatter_robustness.pdf")
