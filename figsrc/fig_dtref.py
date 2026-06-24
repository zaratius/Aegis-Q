"""Fig 4 -- origin of the qubit exit rates. Grouped bars in the shared
shell/funnel colours with 95% Clopper-Pearson whiskers; twin panels share one
rate axis. Counts come ONLY from sim_prototype/eta_sweep_results.json."""
import os, sys, json
sys.path.insert(0, '.')
import numpy as np, matplotlib.pyplot as plt
from scipy import stats
import paper_style as ps; ps.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'sim_prototype', 'eta_sweep_results.json')
if not os.path.exists(DATA):
    raise SystemExit("missing eta_sweep_results.json -- run the eta and dt sweeps first.")
db = json.load(open(DATA))


def cp(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def collect(knob):
    pts = sorted((r for r in db.values() if r['knob'] == knob),
                 key=lambda r: r['value'])
    xs, sh, fn, she, fne = [], [], [], [], []
    for r in pts:
        n = len(r['seeds'])
        if n == 0:
            continue
        ks, kf = sum(r['shell']), sum(r['funnel'])
        xs.append(r['value'])
        sh.append(ks / n); fn.append(kf / n)
        lo, hi = cp(ks, n); she.append((ks / n - lo, hi - ks / n))
        lo, hi = cp(kf, n); fne.append((kf / n - lo, hi - kf / n))
    return xs, np.array(sh), np.array(fn), np.array(she).T, np.array(fne).T


def panel(ax, knob, xlabel, fmt):
    xs, sh, fn, she, fne = collect(knob)
    x = np.arange(len(xs)); w = 0.38
    ek = dict(elinewidth=0.6, capsize=1.5, capthick=0.6, ecolor='0.35')
    ax.bar(x - w / 2, sh, w, color=ps.C_SHELL, label='shell-exit', yerr=she, error_kw=ek)
    ax.bar(x + w / 2, fn, w, color=ps.C_BREACH, label='funnel-exit', yerr=fne, error_kw=ek)
    ax.set_xticks(x); ax.set_xticklabels([fmt(v) for v in xs])
    ax.set_xlabel(xlabel); ax.margins(x=0.04)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(ps.COL, 1.72), sharey=True,
                               constrained_layout=True)
panel(axa, 'eta', r'efficiency $\eta$', lambda v: f'{v:g}')
panel(axb, 'dt', r'step $\Delta t$ (ns)', lambda v: f'{v*1e3:g}')
axa.set_ylabel('exit probability')
axa.text(0.04, 0.93, '(a)', transform=axa.transAxes, fontsize=7)
axb.text(0.04, 0.93, '(b)', transform=axb.transAxes, fontsize=7)
axb.legend(loc='upper right', handlelength=1.1)
fig.savefig('../figures/fig_breach_refinement.pdf')
print("wrote fig_breach_refinement.pdf")
