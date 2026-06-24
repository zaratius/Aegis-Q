"""Fig 5 -- Bell-state stabilization on a shared noise path. Unregularized
(vermillion) vs regularized (green) on three time-locked panels: control,
infidelity (traces coincide), coherent gain. Reproduces the run_bell experiment."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
from quantumdbc import bell, ExpFunnel, SimConfig, run_trajectory

SEED, DT, T = 7, 2.5e-4, 2.0
sys_ = bell(kappa=(50.0, 50.0, 50.0))
ket00 = np.zeros(4, complex); ket00[0] = 1.0; rho0 = np.outer(ket00, ket00.conj())
fun = ExpFunnel(eps0=0.62, eps_T=0.08, T=T)
common = dict(funnel=fun, dt=DT, lam=0.5, s_b=0.05, wgamma=1.0, wdelta=1e3)
tr_un = run_trajectory(sys_, SimConfig(regularized=False, **common), rho0, seed=SEED)
tr_re = run_trajectory(sys_, SimConfig(regularized=True, wr=50.0, c=20.0, **common), rho0, seed=SEED)

CU, CR = ps.C_BREACH, ps.C_FEASIBLE
fig, ax = plt.subplots(3, 1, figsize=(ps.COL, 2.95), sharex=True, constrained_layout=True)
ax[0].axhline(0, color='0.7', lw=0.4)
ax[0].plot(tr_un.t, tr_un.u[:, 0], color=CU, lw=0.45, label='unregularized')
ax[0].plot(tr_re.t, tr_re.u[:, 0], color=CR, lw=1.1, label='regularized')
ax[0].set_ylabel(r'control $u_1$'); ax[0].set_ylim(-1.25, 1.25)
ax[0].legend(loc='upper right', ncol=2, handlelength=1.1, columnspacing=0.8)
ax[0].text(0.015, 0.82, '(a)', transform=ax[0].transAxes, fontsize=7)

ax[1].plot(tr_un.t, tr_un.eps, color=ps.C_FUNNEL, lw=0.8, ls=ps.LS_CONF)
ax[1].plot(tr_un.t, tr_un.xi, color=CU, lw=1.0)
ax[1].plot(tr_re.t, tr_re.xi, color=CR, lw=1.0, ls='--')
ax[1].set_ylabel(r'infidelity $\xi$'); ax[1].set_ylim(0, 0.6)
ax[1].text(0.015, 0.82, r'(b) $\xi$ traces coincide', transform=ax[1].transAxes, fontsize=6.3)

ax[2].plot(tr_un.t, np.abs(tr_un.betaH[:, 0]), color=CU, lw=0.55)
ax[2].plot(tr_re.t, np.abs(tr_re.betaH[:, 0]), color=CR, lw=0.8)
ax[2].set_ylabel(r'gain $|\beta^H_\xi|$'); ax[2].set_ylim(0, None)
ax[2].set_xlabel(r'time $t$ ($\mu$s)'); ax[2].set_xlim(0, tr_un.t.max())
ax[2].text(0.015, 0.82, '(c)', transform=ax[2].transAxes, fontsize=7)
fig.align_ylabels(ax)
fig.savefig('../figures/fig_bell_chatter.pdf'); print("wrote fig_bell_chatter.pdf")
