"""
Quantum Dynamic Barrier Control -- reference simulation.

Reference implementation accompanying Section V of the manuscript.  The
operator data for the qutrit and Bell instances follow the corrected
Section V rewrite (two direct-to-ground dissipators; joint-parity readout).

This code is a reference implementation: results must be validated against
the analytic derivation before use in the paper.
"""
from .systems import System, qubit, qutrit, bell, ggm_generators
from .coefficients import (coefficients_generic, coefficients_closed_form,
                           to_bloch, from_bloch, infidelity)
from .barrier import LogBarrier, FunnelGaugeBarrier, ExpFunnel
from .controller import (QPData, QPResult, solve_closed_form,
                         solve_multichannel, solve_osqp)
from .integrator import milstein_step, euler_step, drift, diffusion
from .simulate import SimConfig, Trajectory, run_trajectory, run_ensemble
from .envelope import qubit_slack_envelope, closed_loop_bound

__all__ = [
    "System", "qubit", "qutrit", "bell", "ggm_generators",
    "coefficients_generic", "coefficients_closed_form",
    "to_bloch", "from_bloch", "infidelity",
    "LogBarrier", "FunnelGaugeBarrier", "ExpFunnel",
    "QPData", "QPResult", "solve_closed_form", "solve_multichannel",
    "solve_osqp",
    "milstein_step", "euler_step", "drift", "diffusion",
    "SimConfig", "Trajectory", "run_trajectory", "run_ensemble",
    "qubit_slack_envelope", "closed_loop_bound",
]
