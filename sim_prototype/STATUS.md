# quantumdbc -- reference simulation: status

Reference implementation accompanying Section V of *Quantum Dynamic Barrier
Control*. Built fresh against the current Section IV/V (temporal
state-dependent weight `w_u(rho)` (the law is Markovian: the temporal
regulariser `w_r` was removed with the carried command), and the corrected
qutrit / Bell operators of the Section V rewrite).

This is a reference implementation. All numerical results must be validated
against the analytic derivation before use in the paper.

## Package layout

    quantumdbc/
      systems.py       su(N) generators; qubit/qutrit/bell operator data
      coefficients.py  generic SME coefficients (7)-(10) + closed-form oracles
      barrier.py       logarithmic barrier; exponential funnel
      controller.py    Theorem IV.2 five-case tree; multichannel root-find; OSQP
      integrator.py    Milstein step; positivity projection
      simulate.py      Algorithm 1 closed loop; Monte Carlo
    tests/             coefficient / controller / integrator verification
    scripts/run_qubit.py  runnable closed-loop demo

## Verification status -- all passing

Coefficients (`tests/test_coefficients.py`)
  - su(N) generators traceless, Tr(T_jT_k)=delta/2, N=2,3,4
  - qubit  Lemma V.1 vs generic SME trace formulas: max dev 2.2e-15
  - qutrit Lemma V.2 vs generic SME trace formulas: max dev 2.0e-15
  - Bell structural facts: sigma(Pi)=0, beta^H(|00>)=0,
    beta^D_k=-kappa_k P_k, separable infidelity >= 1/2
  This verifies the Section V coefficient algebra.

Controller (`tests/test_controller.py`)
  - closed-form (Thm IV.2) vs OSQP, m=p=1: max dev 1.7e-7
  - closed-form vs multichannel root-find: max dev 4.5e-12
  - all five cases exercised
  - multichannel root-find vs OSQP, m=2,p=3: max dev 2.2e-7
  - KKT residual: 8.9e-16

Integrator (`tests/test_integrator.py`)
  - Milstein strong-convergence slope ~1; Euler ~0.5

## Findings -- the Section V-E parameter table needs two changes

1. INTEGRATION STEP. With Omega = 20 the table's Delta t = 1e-3 is too
   coarse: the closed-loop control aliases into a spurious bang-bang limit
   cycle (sign flips on ~98% of steps). At Delta t = 1e-4 the same run is
   smooth (1 sign flip in 50000). Delta t = 1e-4 (or finer) is required for
   the qubit; the E2 chattering figure in particular would be corrupted by
   the artefact at 1e-3. The drive Omega = 20 sets a timescale ~1/20 that
   the step must resolve.

2. FUNNEL FEASIBILITY. The table's qubit funnel (eps0 = 0.40, eps_T = 0.02,
   T = 5) is breached even by a correct, smooth controller: the controller
   drives xi -> 0 but cannot stay inside so aggressive a contraction during
   the transient. A feasible funnel -- e.g. eps0 = 0.55, eps_T = 0.06,
   T = 6, lambda = 0.6 -- gives pathwise confinement with max xi/epsilon
   ~ 0.91. The E3 feasibility check (V(t) >= |eps_dot|) should be run to
   fix the final funnel; the table's funnel and lambda then follow.

A positivity-preserving projection was added to the integrator
(`integrator.project_physical`): finite-step Bloch-vector integration can
leave the physical state space, on which the SME vector field is expansive.
Projection each step (clip negative eigenvalues, renormalise) is a standard
SME-integrator safeguard; the worst excursion is reported per trajectory.

## Remaining work

The three Section V-E figures (E1 qubit confinement + Monte Carlo, E2 Bell
chattering unregularised vs regularised on a shared seed, E3 feasibility)
are not yet generated. They depend on the corrected parameters above; the
funnel and Delta t should be confirmed before the figures are produced, as
they feed the Section V-E table and prose directly.
