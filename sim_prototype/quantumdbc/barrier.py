"""
Barrier function and tolerance funnel (Sections II-D, III-A).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------
# Logarithmic barrier  V(s) = -log(s / s_bar)
# --------------------------------------------------------------------------
class LogBarrier:
    """Canonical barrier of Section III-A satisfying (B1)-(B4).

    V(s)   = -log(s/s_bar)
    V'(s)  = -1/s
    V''(s) =  1/s^2
    kappa_V = -V''/V' = 1/s
    """
    def __init__(self, s_bar: float):
        self.s_bar = float(s_bar)

    def V(self, s: np.ndarray | float):
        return -np.log(np.asarray(s) / self.s_bar)

    def Vp(self, s):
        return -1.0 / np.asarray(s)

    def Vpp(self, s):
        return 1.0 / np.asarray(s) ** 2

    def kappa_V(self, s):
        return 1.0 / np.asarray(s)


# --------------------------------------------------------------------------
# Funnel-gauge barrier  V(s,t) = -log(s / eps(t))
# --------------------------------------------------------------------------
class FunnelGaugeBarrier:
    """Funnel-gauge barrier of the revised Section III-A.

    V(s,t) = -log(s / eps(t)): the tolerance itself sets the scale.  The
    s-derivatives coincide with the fixed-gauge LogBarrier,

        V_s = -1/s,   V_ss = 1/s^2,   zeta_V = 1/s,

    so the KKT reduction is untouched; the explicit time dependence enters
    only through the gauge term  dV/dt|_s = eps_dot/eps <= 0, which cancels
    the contraction cost in the QP drift: the fixed-gauge charge -eps_dot
    in alpha is replaced by -(xi/eps) eps_dot, vanishing at the target.
    Pairs with the relative buffer s_b(t) = theta_b eps(t), so the shell
    sits at the constant barrier level log(1/theta_b).
    """
    def V(self, s: np.ndarray | float, eps_t: np.ndarray | float):
        return -np.log(np.asarray(s) / np.asarray(eps_t))

    def Vp(self, s):
        return -1.0 / np.asarray(s)

    def Vpp(self, s):
        return 1.0 / np.asarray(s) ** 2

    def kappa_V(self, s):
        return 1.0 / np.asarray(s)


# --------------------------------------------------------------------------
# Contracting funnel  epsilon(t)
# --------------------------------------------------------------------------
@dataclass
class ExpFunnel:
    """Exponentially contracting funnel,

        epsilon(t) = eps_T + (eps0 - eps_T) * exp(-r t),

    monotone decreasing with epsilon(0)=eps0 and epsilon(T) -> eps_T.
    The rate r is chosen so that epsilon(T) = eps_T + tol_frac*(eps0-eps_T).
    """
    eps0: float
    eps_T: float
    T: float
    tol_frac: float = 0.05

    def __post_init__(self):
        if self.eps0 <= self.eps_T:
            raise ValueError("require eps0 > eps_T")
        self.r = -np.log(self.tol_frac) / self.T

    def eps(self, t: np.ndarray | float):
        return self.eps_T + (self.eps0 - self.eps_T) * np.exp(-self.r * np.asarray(t))

    def eps_dot(self, t: np.ndarray | float):
        return -self.r * (self.eps0 - self.eps_T) * np.exp(-self.r * np.asarray(t))
