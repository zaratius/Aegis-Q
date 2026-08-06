"""Fig 6 -- unregularized control sign-flip fraction vs integration step, qubit
and Bell, averaged over SEEDS. Bell rises to a plateau ~0.9 (gain at the exact
obstruction zero: sign redrawn by the innovation each step, dt-independent).
The qubit fraction is non-monotone, staying in [0.4, 0.65] -- it does NOT
plateau, but neither does it fall like an O(dt) discretization artifact.
Seed-averaged values (8 seeds): qubit [0.581 0.609 0.651 0.405],
bell [0.617 0.873 0.901 0.911] for dts (2e-3, 1e-3, 5e-4, 2.5e-4).

NOTE: the paper now presents these numbers as Table III (tab:signflip) --
the printed line below is the table's data source, transcribed by hand.
The PDF this script renders is no longer included in the paper; if SEEDS
or dts change, update the table in Chapters/Applications.tex to match."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
from quantumdbc import qubit, bell, ExpFunnel, SimConfig, run_trajectory

SEEDS = range(7, 15); dts = (2e-3, 1e-3, 5e-4, 2.5e-4)
sq = qubit(); rq = np.array([[0.78, 0.08], [0.08, 0.22]], complex)
fq = ExpFunnel(eps0=0.55, eps_T=0.10, T=1.0)
sb = bell(); ket = np.zeros(4, complex); ket[0] = 1.0; rb = np.outer(ket, ket.conj())
fb = ExpFunnel(eps0=0.62, eps_T=0.20, T=1.0)
ffq, ffb = [], []
for dt in dts:
    cq = SimConfig(funnel=fq, dt=dt, lam=0.6, theta_b=0.10, regularized=False)
    cb = SimConfig(funnel=fb, dt=dt, lam=0.5, theta_b=0.10, regularized=False)
    ffq.append(np.mean([np.mean(np.diff(np.sign(run_trajectory(sq, cq, rq, seed=s).u[:, 0])) != 0)
                        for s in SEEDS]))
    ffb.append(np.mean([np.mean(np.diff(np.sign(run_trajectory(sb, cb, rb, seed=s).u[:, 0])) != 0)
                        for s in SEEDS]))
print("seed-averaged flip fractions  qubit:", np.round(ffq, 3), " bell:", np.round(ffb, 3))
dns = np.asarray(dts) * 1e4
fig, ax = plt.subplots(figsize=(ps.COL, 1.60), constrained_layout=True)
ax.semilogx(dns, ffq, 'o-', color="#000000", lw=0.5, ms=2.5, label='qubit')
ax.semilogx(dns, ffb, 's-', color="#000000", lw=0.5, ms=2.5, label=r'Bell $|00\rangle$')
ax.set_xlabel(r'integration step $\Delta t$ ($10^{-4}\,\Gamma_m^{-1}$)'); ax.set_ylabel('sign-flip fraction')
ax.set_ylim(0, 1); ax.invert_xaxis()
ax.legend(loc='lower left', handlelength=1.3)
ax.annotate(r'plateau as $\Delta t\!\to\!0$', xy=(dns[-1], ffb[-1] - 0.14),
            xytext=(dns[1], 0.28), fontsize=6, color='0.45',
            arrowprops=dict(arrowstyle='->', color='0.5', lw=0.5))
fig.savefig('../figures/fig_chatter_robustness.pdf'); print("wrote fig_chatter_robustness.pdf")
