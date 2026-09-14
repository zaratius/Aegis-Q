#Fig 3 -- funnel feasibility on the design-shell boundary
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
from quantumdbc.systems import qubit
from quantumdbc.coefficients import coefficients_generic
from quantumdbc.barrier import ExpFunnel

sys_ = qubit(); fun = ExpFunnel(eps0=0.70, eps_T=0.0001, T=4.0)
umax, gmax = 1.0, 3.90            
TB_ADM, TB_TIGHT = 0.20, 0.07     


def speed_limit_at_boundary(t, theta_b):
    eps = float(fun.eps(t)); epsdot = float(fun.eps_dot(t))
    xi = (1.0 - theta_b) * eps
    zeta = 1.0 / (theta_b * eps)
    pref = eps / xi                        
    best = np.inf
    for c2 in np.linspace(-0.45, 0.45, 60):
        r = np.array([[1 - xi, c2 / 2], [c2 / 2, xi]], dtype=complex)
        if np.linalg.eigvalsh(r).min() < -1e-9:
            continue
        co = coefficients_generic(sys_, r)
        bH = float(np.atleast_1d(co["betaH"])[0]); bD = float(np.atleast_1d(co["betaD"])[0])
        best = min(best, pref * float(-co["mu"] - 0.5 * zeta * co["sigma"] ** 2
                                      + abs(bH) * umax + abs(bD) * gmax))
    return best, abs(epsdot)


ts = np.linspace(0.02, fun.T, 80)
Vadm, dem, Vtight = [], [], []
for t in ts:
    v, d = speed_limit_at_boundary(t, TB_ADM); Vadm.append(v); dem.append(d)
    Vtight.append(speed_limit_at_boundary(t, TB_TIGHT)[0])
Vadm, dem, Vtight = map(np.array, (Vadm, dem, Vtight))
np.savez('feas_data.npz', ts=ts, Vadm=Vadm, dem=dem, Vtight=Vtight,
         theta_adm=TB_ADM, theta_tight=TB_TIGHT)

fig, ax = plt.subplots(figsize=(ps.COL, 1.48), constrained_layout=True)
ax.fill_between(ts, dem, Vadm, where=(Vadm >= dem), color=ps.C_FEASIBLE, alpha=0.13, lw=0)
ax.plot(ts, Vadm, color="#000000", lw=0.5, label=r'speed limit $\mathcal{V}(t)$')
ax.plot(ts, dem, color="#000000", lw=0.5, ls='--', label=r"demand $|\dot\epsilon|$")
ax.set_xlabel(r'time $t$ ($\Gamma_m^{-1}$)'); ax.set_ylabel('rate'); ax.set_xlim(0, fun.T)
ax.legend(loc='upper right', handlelength=1.4)
fig.savefig('../figures/fig_feasibility.pdf'); plt.close(fig)

fig, ax = plt.subplots(figsize=(ps.COL, 1.48), constrained_layout=True)
madm, mtight = Vadm - dem, Vtight - dem
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.fill_between(ts, mtight, 0, where=(mtight < 0), color="#BF0606", alpha=0.13, lw=0)
ax.plot(ts, madm, color="#000000", lw=0.5, label=rf'$\theta_b{{=}}{TB_ADM}$ (admissible)')
ax.plot(ts, mtight, color="#000000", lw=0.5, ls='--', label=rf'$\theta_b{{=}}{TB_TIGHT}$ (inadmissible)')
ax.set_xlabel(r'time $t$ ($\Gamma_m^{-1}$)'); ax.set_ylabel(r'margin $\mathcal{V}-|\dot\epsilon|$')
ax.set_xlim(0, fun.T); ax.legend(loc='center right', handlelength=1.4)
fig.savefig('../figures/fig_feasibility_small_buffer.pdf'); plt.close(fig)
print("wrote fig_feasibility.pdf + fig_feasibility_small_buffer.pdf")
