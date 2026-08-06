"""Post-hoc exit metrics for a stored closed-loop trajectory.

Computes the two distinct exit events of the funnel-gauge Section III theory,
plus the running-max overshoot (the plateau-vs-decay disambiguator) and the
V-space accumulated slack (the Corollary III.2 quantity).

Works entirely from a stored trajectory (run with store=True); it does not
touch run_trajectory internals, so it is safe to bolt onto existing scripts.

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

Why gate the slack at s >= theta_b * eps
----------------------------------------
  The V-space slack is delta_V = (-V_s) delta_QP = delta_QP / s. The 1/s
  factor diverges as s -> 0. Corollary III.2's shell bound applies on
  [0, tau_Omega), where s >= theta_b eps(t) >= theta_b eps(T) keeps Delta(T)
  finite. Accumulating past tau_Omega toward s -> 0 diverges -- and that
  divergence is precisely WHY Doob does not forbid funnel-exits -- so it must
  not leak into the shell bound. Hence the gate.
"""

import numpy as np


def exit_metrics(tr, theta_b):
    """Return a dict of exit metrics for one stored trajectory.

    Parameters
    ----------
    tr      : trajectory with .xi, .eps, .t, .delta arrays (store=True)
    theta_b : relative buffer (same value used in the SimConfig)

    Returns
    -------
    dict with keys:
      tau_Omega, tau_aleph : first-passage times (np.inf if no exit on [0,T])
      shell_exit, funnel_exit : bool, whether each event occurred on [0,T]
      runmax  : max_t xi/eps  (overshoot; runmax-1 is the scaling observable)
      Delta_T : V-space accumulated slack, sum(delta/s) dt gated on the shell
      V_level : the constant barrier level log(1/theta_b) the shell sits at
    """
    xi  = np.asarray(tr.xi, dtype=float)
    eps = np.asarray(tr.eps, dtype=float)
    t   = np.asarray(tr.t, dtype=float)

    s = eps - xi                                  # TRUE margin (not floored)
    sb = theta_b * eps                            # relative buffer width
    if t.size > 1:
        dt = float(np.median(np.diff(t)))
    else:
        dt = 0.0

    below_shell = s < sb
    exited      = s <= 0.0

    # argmax on a boolean array returns the FIRST True (first-passage), or 0
    # if all False -- so guard with .any().
    tau_Omega = float(t[np.argmax(below_shell)]) if below_shell.any() else np.inf
    tau_aleph = float(t[np.argmax(exited)])       if exited.any()      else np.inf

    runmax = float(np.max(xi / eps))

    # V-space slack, gated to the shell so the 1/s factor stays bounded.
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
    """Fit mean overshoot (mean of runmax-1) against sqrt(dt ln(T/dt)).

    A near-boundary diffusion's running-max overshoot over a fixed horizon
    scales like sqrt(dt ln(T/dt)). Fitting that to the data separates the
    two hypotheses by the INTERCEPT:

      intercept ~ 0  -> finite Delta(T), continuum funnel-exit prob 0:
                        overshoots are finite-dt and vanish under refinement
                        (Framing B / O(dt) artifact).
      intercept > 0  -> genuine continuum overshoot, positive exit prob
                        (Framing A / Doob-bounded noise excursion).

    Returns (slope, intercept).
    """
    dts = np.asarray(dts, dtype=float)
    y   = np.asarray(mean_overshoots, dtype=float)
    x   = np.sqrt(dts * np.log(T / dts))
    A   = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)
