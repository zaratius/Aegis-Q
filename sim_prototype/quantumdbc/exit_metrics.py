"""Post-hoc exit metrics for a stored closed-loop trajectory.

Computes the two distinct exit events of the funnel-gauge Section III theory

Works entirely from a stored trajectory (run with store=True); it does not
touch run_trajectory internals.

The two events
--------------
  tau_Omega : first time the TRUE margin s = eps - xi drops below the
              relative buffer theta_b * eps(t). The path has left the design
              shell into the buffer layer but is STILL INSIDE the funnel:
              design-margin erosion, NOT a safety failure. V crosses the
              constant level log(1/theta_b) -- constant because buffer and
              barrier share the gauge -- so this event carries a positive,
              finite Doob bound (Corollary III.2).

  tau_aleph : first time s <= 0, i.e. xi >= eps. This is a true funnel exit,
              the safety-critical event. V -> +infty here. With finite V-space
              Delta(T) the continuum probability of this event is ZERO
              (Theorem III.1) -- so a persistent tau_aleph rate under
              refinement means the supermartingale is FAILING (funnel
              infeasible on the realized shell), not "noise bounded by Doob".

"""

import numpy as np


def exit_metrics(tr, theta_b):
    xi  = np.asarray(tr.xi, dtype=float)
    eps = np.asarray(tr.eps, dtype=float)
    t   = np.asarray(tr.t, dtype=float)

    s = eps - xi                                
    sb = theta_b * eps                            
    if t.size > 1:
        dt = float(np.median(np.diff(t)))
    else:
        dt = 0.0

    below_shell = s < sb
    exited      = s <= 0.0


    tau_Omega = float(t[np.argmax(below_shell)]) if below_shell.any() else np.inf
    tau_aleph = float(t[np.argmax(exited)])       if exited.any()      else np.inf

    runmax = float(np.max(xi / eps))

    delta = np.asarray(getattr(tr, "delta", np.zeros_like(t)), dtype=float)
    gate  = s >= sb
    if gate.any():
        Delta_T = float(np.sum(delta[gate] / s[gate]) * dt)
    else:
        Delta_T = 0.0

    V_level = float(np.log(1.0 / theta_b))

    return dict(
        tau_Omega=tau_Omega,
        tau_aleph=tau_aleph,
        shell_exit=bool(tau_Omega < np.inf),
        funnel_exit=bool(tau_aleph < np.inf),
        runmax=runmax,
        Delta_T=Delta_T,
        V_level=V_level,
    )


def overshoot_scaling(dts, mean_overshoots, T):
    dts = np.asarray(dts, dtype=float)
    y   = np.asarray(mean_overshoots, dtype=float)
    x   = np.sqrt(dts * np.log(T / dts))
    A   = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)
