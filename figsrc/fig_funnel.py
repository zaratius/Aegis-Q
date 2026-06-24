"""Fig 2 -- funnel geometry, drawn against real (error, time) axes in the same
vocabulary as the data figures. Illustrative trajectory; geometry only."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
from quantumdbc.barrier import ExpFunnel

fun = ExpFunnel(eps0=0.62, eps_T=0.13, T=4.0); s_b = 0.08
t = np.linspace(0, 4, 600); eps = fun.eps(t); shell = eps - s_b

# illustrative path: decay + smoothed noise + two crafted excursions
rng = np.random.default_rng(4)
sm = np.convolve(rng.normal(0, 1, t.size), np.ones(16) / 16, "same")
xi = (0.255 * np.exp(-0.5 * t) + 0.035 * sm
      + 0.150 * np.exp(-((t - 1.30) / 0.15) ** 2)     # dip into the collar (~1.3)
      + 0.150 * np.exp(-((t - 2.50) / 0.18) ** 2))    # over the funnel (~2.5)
xi = np.clip(xi, 0.004, None)
i_sh = np.argmax(xi > shell); i_fn = np.argmax(xi >= eps)

fig, ax = plt.subplots(figsize=(ps.COL, 1.72), constrained_layout=True)
ax.fill_between(t, 0, shell, color=ps.C_FEASIBLE, alpha=0.10, lw=0)         # interior
ax.fill_between(t, shell, eps, color=ps.C_SHELL, alpha=0.10, lw=0)          # buffer collar
ax.plot(t, eps, color=ps.C_FUNNEL, lw=1.2)                                  # funnel boundary
ax.plot(t, shell, color=ps.C_SHELL, lw=1.0, ls=ps.LS_SHELL)                 # design shell
ax.plot(t, xi, color=ps.C_CONFINED, lw=0.8, ls=ps.LS_CONF)                  # trajectory
ax.plot(t[i_sh], xi[i_sh], 'o', color=ps.C_SHELL, ms=4, zorder=8)
ax.plot(t[i_fn], xi[i_fn], 'o', color=ps.C_BREACH, ms=4, zorder=8)
ax.annotate(r'funnel $\xi=\epsilon(t)$', (3.0, eps[450]), (2.4, 0.30),
            fontsize=6, color=ps.C_FUNNEL)
ax.annotate(r'shell $\Omega(t)$', (3.3, shell[495]), (2.7, 0.015),
            fontsize=6, color=ps.C_SHELL)
ax.text(0.12, 0.045, r'interior $\aleph(t)$ -- confined', fontsize=6,
        color=ps.C_FEASIBLE)
ax.annotate(r'shell-exit $\tau_\Omega$', (t[i_sh], xi[i_sh]), (0.7, 0.40),
            fontsize=6, color=ps.C_SHELL,
            arrowprops=dict(arrowstyle='-', color=ps.C_SHELL, lw=0.4))
ax.annotate(r'funnel-exit $\tau_\aleph$', (t[i_fn], xi[i_fn]), (2.55, 0.50),
            fontsize=6, color=ps.C_BREACH,
            arrowprops=dict(arrowstyle='-', color=ps.C_BREACH, lw=0.4))
ax.set_xlim(0, 4); ax.set_ylim(0, 0.72)
ax.set_xlabel(r'time $t$ ($\mu$s)'); ax.set_ylabel(r'error $\xi$')
ax.grid(True, axis='both')
fig.savefig('../figures/fig_funnel.pdf'); print("wrote fig_funnel.pdf")
