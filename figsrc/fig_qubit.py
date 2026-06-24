"""Fig 3 -- qubit ground-state confinement. Confined paths recede to thin grey
dotted; shell- and funnel-exits read instantly in blue/vermillion over the
shared funnel/shell reference. Lower strip: pointwise confinement frequency."""
import sys; sys.path.insert(0, '.')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps; ps.apply()

d = np.load('qubit_data.npz'); t = d['t']; eps = d['eps']; xis = d['xis']
s_b = float(d['s_b']); M = len(xis)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ps.COL, 2.30),
                               height_ratios=[2.0, 1.0], sharex=True,
                               constrained_layout=True)
ps.funnel_curves(ax1, t, eps, s_b, fill=True)
nshow = min(28, M)
order = np.argsort([np.max(x / eps) for x in xis])[::-1]      # worst on top
for i in order[:nshow][::-1]:
    x = xis[i]
    if np.any(x >= eps):       c, a, z, lw, ls = ps.C_BREACH, 0.95, 5, 0.6, '-'
    elif np.any(x > eps - s_b): c, a, z, lw, ls = ps.C_SHELL, 0.75, 4, 0.5, '-'
    else:                       c, a, z, lw, ls = ps.C_CONFINED, 0.45, 2, 0.4, ps.LS_CONF
    ax1.plot(t, x, color=c, lw=lw, alpha=a, zorder=z, ls=ls, rasterized=True)
ax1.annotate(r'$\epsilon(t)$', (t[int(.62*len(t))], eps[int(.62*len(t))]),
             (t[int(.62*len(t))], eps[int(.62*len(t))]+0.06), fontsize=6)
proxies = [Line2D([0], [0], color=ps.C_CONFINED, lw=0.9, ls=ps.LS_CONF),
           Line2D([0], [0], color=ps.C_SHELL, lw=1.0),
           Line2D([0], [0], color=ps.C_BREACH, lw=1.0)]
ax1.legend(proxies, ['confined', 'shell-exit', 'funnel-exit'],
           loc='upper right', ncol=1, handlelength=1.4)
ax1.set_ylabel(r'infidelity $\xi(t)$'); ax1.set_ylim(0, 0.74)
ax1.set_xlim(0, t.max())

conf = 100 * np.mean(xis < eps[None, :], axis=0)
ax2.plot(t, conf, color=ps.C_FEASIBLE, lw=1.1)
ax2.set_ylim(max(88, conf.min() - 1), 100.5)
ax2.set_xlabel(r'time $t$ ($\mu$s)'); ax2.set_ylabel(r'confined (\%)')
ax2.text(0.985, 0.12, rf'$M={M}$', transform=ax2.transAxes, ha='right',
         fontsize=6, color='0.45')
fig.savefig('../figures/fig_qubit_confinement.pdf', dpi=600)
print(f"wrote fig_qubit_confinement.pdf  (final {conf[-1]:.1f}%, min {conf.min():.1f}%)")
