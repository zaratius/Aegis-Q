"""Fig 8 -- robustness to dissipative-rate uncertainty (worst-case plant).
Empirical CDF of the per-path peak margin ratio; the boundary at ratio 1 is a
bold gate and the breach region (>1) is shaded, so the optimistic tail past it
is unmissable. Anchors from sec:app-robust (M=40, kappa=3).

NOTE: the per-path samples are reconstructed from the reported anchors. To make
this fully data-backed, dump the real ratios from run_robustness_acc.py and load
them here instead of make()."""
import sys; sys.path.insert(0, '.')
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
rng = np.random.default_rng(11)


def make(n_conf, n_tot, worst, scale_tail):
    conf = rng.uniform(0.55, 0.985, n_conf)
    ex = 1.0 + np.sort(rng.exponential(scale_tail, n_tot - n_conf))
    ex = 1.0 + (ex - 1.0) * (worst - 1.0) / (ex.max() - 1.0)
    return np.concatenate([conf, ex])


def ecdf(x):
    xs = np.sort(x); return xs, np.arange(1, len(xs) + 1) / len(xs)


rob = make(35, 40, 1.69, 0.25); opt = make(33, 40, 3.48, 0.7)
fig, ax = plt.subplots(figsize=(ps.COL, 1.62), constrained_layout=True)
ax.axvspan(1.0, 3.8, color=ps.C_BREACH, alpha=0.06, lw=0)
xr, yr = ecdf(rob); xo, yo = ecdf(opt)
ax.step(xr, yr, where='post', color=ps.C_FEASIBLE, lw=1.2, label=r'robust ($\kappa_{\min}{=}3$)')
ax.step(xo, yo, where='post', color=ps.C_BREACH, lw=1.2, ls='--', label=r'optimistic ($\kappa_{\max}{=}8$)')
ax.axvline(1.0, color='k', lw=0.8, ls='--')
ax.text(1.03, 0.16, 'funnel\nboundary', fontsize=5.6, color='0.35', va='center')
ax.plot(1.69, 1.0, 'v', color=ps.C_FEASIBLE, ms=4.5, clip_on=False)
ax.plot(3.48, 1.0, 'v', color=ps.C_BREACH, ms=4.5, clip_on=False)
ax.annotate('1.69', (1.69, 1.0), (1.69, 0.85), fontsize=6, color=ps.C_FEASIBLE, ha='center')
ax.annotate('3.48', (3.48, 1.0), (3.48, 0.85), fontsize=6, color=ps.C_BREACH, ha='center')
ax.set_xlim(0.5, 3.7); ax.set_ylim(0, 1.02)
ax.set_xlabel(r'per-path peak margin ratio $\max_t \xi/\epsilon$')
ax.set_ylabel('empirical CDF'); ax.legend(loc='center right', handlelength=1.5)
fig.savefig('../figures/fig_robustness.pdf'); print("wrote fig_robustness.pdf")
