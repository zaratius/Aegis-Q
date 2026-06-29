"""Fig 6 -- funnel feasibility on the design-shell boundary. (a) contraction demand
|eps_dot| against the speed limit V(t) for the admissible buffer, green headroom
filled; (b) feasibility margin V-|eps_dot| for two buffers, zero line bold so the
inadmissible dip below it is the obvious event."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_prototype'))
import numpy as np, matplotlib.pyplot as plt
import paper_style as ps; ps.apply()
from quantumdbc.systems import qubit
from quantumdbc.coefficients import coefficients_generic
from quantumdbc.barrier import ExpFunnel

sys_ = qubit(); fun = ExpFunnel(eps0=0.70, eps_T=0.30, T=4.0)
umax = gmax = 1.0


def speed_limit_at_boundary(t, s_b):
    # Evaluate on the design-shell boundary dOmega(t) = {xi = eps(t) - s_b}, where
    # the barrier is active and the headroom to the funnel equals the buffer s_b.
    eps = float(fun.eps(t)); epsdot = float(fun.eps_dot(t))
    xi = eps - s_b; zeta = 1.0 / s_b; best = np.inf
    for c2 in np.linspace(-0.45, 0.45, 60):
        r = np.array([[1 - xi, c2 / 2], [c2 / 2, xi]], dtype=complex)
        if np.linalg.eigvalsh(r).min() < -1e-9:
            continue
        co = coefficients_generic(sys_, r)
        bH = float(np.atleast_1d(co["betaH"])[0]); bD = float(np.atleast_1d(co["betaD"])[0])
        best = min(best, float(-co["mu"] - 0.5 * zeta * co["sigma"] ** 2
                               + abs(bH) * umax + abs(bD) * gmax))
    return best, abs(epsdot)


ts = np.linspace(0.02, fun.T, 80)
V25, dem, V05 = [], [], []
for t in ts:
    v, d = speed_limit_at_boundary(t, 0.25); V25.append(v); dem.append(d)
    V05.append(speed_limit_at_boundary(t, 0.05)[0])
V25, dem, V05 = map(np.array, (V25, dem, V05))
np.savez('feas_data.npz', ts=ts, V25=V25, dem=dem, V05=V05)

fig, ax = plt.subplots(figsize=(ps.COL, 1.48), constrained_layout=True)
ax.fill_between(ts, dem, V25, where=(V25 >= dem), color=ps.C_FEASIBLE, alpha=0.13, lw=0)
ax.plot(ts, V25, color="#000000", lw=0.5, label=r'speed limit $\mathcal{V}(t)$')
ax.plot(ts, dem, color="#000000", lw=0.5, ls='--', label=r"demand $|\dot\epsilon|$")
ax.set_xlabel(r'time $t$ ($\mu$s)'); ax.set_ylabel('rate'); ax.set_xlim(0, fun.T)
ax.legend(loc='upper right', handlelength=1.4)
fig.savefig('../figures/fig_feasibility.pdf'); plt.close(fig)

fig, ax = plt.subplots(figsize=(ps.COL, 1.48), constrained_layout=True)
m25, m05 = V25 - dem, V05 - dem
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.fill_between(ts, m05, 0, where=(m05 < 0), color="#BF0606", alpha=0.13, lw=0)
ax.plot(ts, m25, color="#000000", lw=0.5, label=r'$s_b{=}0.25$ (admissible)')
ax.plot(ts, m05, color="#000000", lw=0.5, ls='--', label=r'$s_b{=}0.05$ (inadmissible)')
ax.set_xlabel(r'time $t$ ($\mu$s)'); ax.set_ylabel(r'margin $\mathcal{V}-|\dot\epsilon|$')
ax.set_xlim(0, fun.T); ax.legend(loc='center right', handlelength=1.4)
fig.savefig('../figures/fig_feasibility_small_buffer.pdf'); plt.close(fig)
print("wrote fig_feasibility.pdf + fig_feasibility_small_buffer.pdf")
