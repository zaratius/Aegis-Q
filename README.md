# Aegis — reference implementation for *Safety-Critical Control of Continuously Measured Open Quantum Systems via Barrier Functions*

Closed-loop simulation of the barrier-function controller (Algorithm 1 of the paper), the a priori shell-exit bound (Corollary IV.3), and the scripts that produce every data figure in Section VI.

## Layout

```
sim_prototype/quantumdbc/   library: SME coefficients, barrier, QP solvers, Milstein integrator, envelope
sim_prototype/test/         verification of the coefficient algebra, the QP solvers, and the integrator order
sim_prototype/run_*.py      Monte Carlo studies; write results to sim_prototype/*.json
figsrc/                     figure scripts; read the JSON/NPZ results and write figures/*.pdf
figures/                    the PDFs used in the paper
regen_figures.sh            runs the whole pipeline in order
```

## Figure provenance

| Paper | File | Produced by |
|---|---|---|
| Fig. 3 | `figures/fig_feasibility_small_buffer.pdf` | `figsrc/fig_feas.py` |
| Fig. 4 | `figures/fig_qubit_confinement.pdf` | `sim_prototype/run_e1_acc.py` |
| Fig. 5 | `figures/fig_breach_refinement.pdf` | `figsrc/fig_dtref.py` ← `sim_prototype/eta_sweep_results.json` ← `run_eta_sweep.py` |
| Fig. 6 | `figures/fig_robustness.pdf` | `sim_prototype/run_robustness_acc.py` → also `figures/robustness_excursions.npz` |
| Fig. 7 | `figures/fig_bell_confinement.pdf` | `sim_prototype/run_bell_mc.py` ← `sim_prototype/bell_mc_results.json` |

Figs. 1 and 2 are drawn in TikZ in the manuscript.

The certified bounds quoted in the paper come from the envelope computations:
qubit 0.239 (`run_e1_acc.py`, grid 161×321×161) and Bell 0.568 (`run_bell_mc.py envelope`).

## Parameters

All design parameters (funnel, buffer θ_b, actuator bounds, QP weights, initial states) live in one place, `sim_prototype/quantumdbc/study_config.py`, and match Table II of the paper. The file asserts the design conditions at import time (shell-edge feasibility, the initial state inside the shell), so an inconsistent edit fails immediately.

## Requirements

Python ≥ 3.11 with `numpy`, `scipy`, `matplotlib`, `scienceplots`, `numba`. `osqp` is optional and only used by the controller cross-check in `test/test_controller.py`.

```
pip install numpy scipy matplotlib scienceplots numba osqp
```

## Reproducing

```
bash regen_figures.sh            # all figures from the stored sweep data (~2 min)
bash regen_figures.sh --sweeps   # additionally re-run the η / Δt sweeps (slow)
```

Individual pieces:

```
cd sim_prototype
python run_e1_acc.py                  # qubit confinement, M = 1000, and the qubit envelope bound
python run_bell_mc.py envelope        # Bell certified bound (refinement ladder)
python run_bell_mc.py 1000            # Bell ensemble, M = 1000
python run_robustness_acc.py 1000     # robustness study, M = 1000 per design
python run_eta_sweep.py eta 300 0     # η sweep, 300 trajectories per point
python run_eta_sweep.py dt  300 0     # Δt sweep
cd ../figsrc
python fig_feas.py                    # feasibility margin
python fig_dtref.py                   # exit rates vs η and Δt
```

Wiener increments are seeded per trajectory, so every run above is reproducible bit-for-bit under the same seed base.

## Tests

```
cd sim_prototype
python -m pytest test/
```

Checks: the closed-form coefficients (55) and Lemmas V.4–V.5 against the generic SME trace formulas; the five-case law (Theorem IV.2) and the multi-channel root find against each other and against OSQP; the Milstein strong-order slope; and the Numba kernel against the pure-Python reference on identical noise.

## Notes

- The Bell instance uses the multi-channel root find (`controller.solve_multichannel`), not the five-case tree; the tree is exact for m = p = 1 only.
- `envelope.py` restricts the Bell supremum to the Bell-diagonal simplex; the docstring gives the argument that this loses nothing.
- Runs use a floored margin `s ← max(ε − ξ, θ_b ε)` (Algorithm 1, line 4), which is inactive before the first shell exit.
