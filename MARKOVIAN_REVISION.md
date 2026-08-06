# Markovian-law code revision — numbers memo (2026-08-06)

Code in this repo now implements the Markovian barrier law of the revised
manuscript: no temporal weight `w_r`, no carried command `u_prev`. Commits:
`34c210b` (pre-refactor checkpoint) → `9ec1cf3` (refactor). SCCQS was not
touched (read-only session).

## 1. Verification (all on this box, pure-python path)

| Check | Result |
|---|---|
| 10,000 random single-channel QPs, old(w_r=0) vs new closed form | max deviation **exactly 0.0**, all case labels identical |
| 10,000 random multichannel QPs (m,p ≤ 3), old(w_r=0) vs new root find | max deviation **exactly 0.0** |
| Paired qubit trajectory, paper config, seed 7 (8000 steps) | all recorded arrays **bitwise identical** |
| Paired Bell trajectory, fig-E2 config, seed 7 (8000 steps) | all recorded arrays **bitwise identical** |
| test_coefficients.py, test_integrator.py | pass |
| test_controller.py (needs osqp), test_backends.py (needs numba) | **run on the Mac** — see §6 |

The 2026-07-28 ablation already established that w_r=0 leaves the paired
MC ensembles unchanged (max |Δ(ξ/ε)| < 5e-7 over M=100, 0/100
classification changes), so **no Monte-Carlo figures change**: figs E1,
deadline, robustness, feasibility, Δt-refinement all stand as rendered.

## 2. Certified qubit bound — replaces 0.55 in Section V

Envelope grid sup over Ω(t) only (the carried-command axis is gone):

| grid (t × x3 × x2) | Δ = ∫δ̄_V dt | bound |
|---|---|---|
| 161 × 321 × 161 (paper grid) | 0.000130 | 0.359513 |
| 241 × 481 × 241 | 0.000511 | 0.359876 |
| 321 × 641 × 321 | 0.000555 | 0.359918 |
| 481 × 961 × 481 | 0.000550 | 0.359912 |

- **P[τ_Ω ≤ T] ≤ 0.360** (converged; stable to 4e-5 over the last three
  levels). Doob floor V₀/log(1/θ_b) = 0.37729/1.04982 = **0.35939**.
- The envelope contribution is Δ/level ≈ 5×10⁻⁴: with the history bias
  gone, the certified bound **coincides with the slack-free Doob floor to
  three decimals**. w_r was generating essentially all of the old slack
  (0.55 → 0.36).
- max δ̄_V(t) ≈ 3.0×10⁻⁴ (attained near the horizon).

### Suggested Section-V text updates (SCCQS is read-only for me)
- Applications.tex:464 — replace the provenance line: shell grid is now
  "161×321×161 in t×x3×x2" **with no "five carried commands"** clause;
  suggest quoting the finest grid or adding "stable over a four-level
  refinement ladder".
- Applications.tex:466 — "gives P[τ_Ω≤T] ≤ 0.55" → **"≤ 0.36"**.
- Applications.tex:474–484 — the margin discussion changes character:
  observed shell-exit 25.3% [20.5, 30.7] still sits under the bound, but
  the bound now *equals* the Doob floor, so "below even the slack-free
  Doob floor" should become e.g. "the certified bound coincides with the
  slack-free Doob floor (the finite-w_δ slack contributes 5×10⁻⁴); the
  observed rate sits under it with a factor ≈1.4 margin". Delete "the
  worst carried command" at :480.
- Applications.tex:498–504 (deadline) — 1 − 0.55 = 0.45 becomes
  **1 − 0.36 = 0.64**; the certified statement strengthens to
  P[ξ(t) ≤ (1−θ_b)ε(t) ∀t] ≥ 0.64. ξ(T) ≤ (1−θ_b)ε(T) = 0.65 × 0.32 =
  0.208 → the "ξ(T) ≤ 0.21" claim is unchanged. Empirical 74.7% / 94.0% /
  92.7% comparisons unchanged.
- Table (Applications.tex:427) — delete the "Temporal weight w_r | 50 | 50"
  row.
- Applications.tex:547 — "(w_r=0, w_u≡1)" → "(w_u ≡ 1)".

## 3. Bell E2 statistics under the new law (fig config, seed 7)

κ = 50, ε: 0.62 → 0.08, T = 2, Δt = 2.5e-4, shared noise path:

| law | flip fraction | mean per-step |du| | mean |u₁| | ξ(0) → ξ(T) |
|---|---|---|---|---|
| regularized (w_u = c/(\|β^H\|+ε_f), c = 20) | 0.008 | 0.0000 | 0.0000 | 0.5000 → 0.0370 |
| unregularized (w_u ≡ 1) | 0.294 | 0.456 | 0.320 | 0.5000 → 0.0368 |

The E2 narrative survives w_r removal intact: the state-dependent weight
alone suppresses the chattering (mean coherent effort ≈ 0) and the two
infidelity endpoints coincide. `fig_bell_chatter` needs a Mac re-render
(only figure whose generating config changed); expect it visually
indistinguishable from the current PDF.

## 4. Bell δ̄_V — exploratory diagnostic verdict

- **The certificate is structurally silent for the E2 study as designed,
  independent of any envelope**: ρ₀ = |00⟩⟨00| has ξ₀ = 1/2 >
  (1−θ_b)ε(0) = 0.434 — the path starts *outside the design shell*, and
  V₀/log(1/θ_b) = 1.6422/1.2040 = **1.36 > 1**. Keeping the paper's
  "certificate scoped to the qubit" is forced, not just prudent.
- Analytic upper bound (ν* ≤ w_δΘ₊ route, ignores actuators): vacuous by
  ~3 orders (Δ ≈ 897). Not usable.
- Sampled maximization (41 t-points × 300 Ginibre × 9 exact-ξ levels +
  300-step stochastic polish, real multichannel solver): the objective was
  **≤ 0 at every sampled shell state** — δ̄_V ≡ 0 to sampling resolution.
  The three κ=50 dissipators cover the constraint everywhere probed; the
  dissipative stiffness (β^D)²/w_γ cannot vanish on the shell (Σ_k
  ⟨ψ_k|ρ|ψ_k⟩ = ξ > 0 there). Caveat: a lower estimate from typical
  (full-rank) states; adversarial pure states were probed only via the
  polish.
- **If a certified Bell instance is ever wanted**: the envelope is not the
  obstacle — the Doob floor is. With δ̄_V ≈ 0 a bound of 0.9 needs
  ξ₀ ≤ 0.41; a 0.36-style bound needs ξ₀ ≈ 0.32. That means preparing an
  initial state well inside the shell (e.g. by running the loop first and
  certifying only a later segment), plus a grid-free envelope argument.
  Recommendation: leave the paper's scoping unchanged; optionally add one
  sentence noting the Bell certificate fails at the initial condition, not
  at the envelope.

## 5. Stale artifacts remaining in SCCQS (read-only for me — your edit pass)

- `Chapters/Algo_revise.tex`: caption "(regularized)" (:2); `w_r` in
  \Require (:8); "Regularization weights" comment (:21); `\weq ← w_u+w_r`
  (:22); `Θ ← α + β^H w_r u_-/\weq + λV` (:24) → should be `Θ ← α + λV`;
  dormant `u* ← w_r u_-/\weq` (:26) → `u* ← 0`.
- `Chapters/Optimization_revised.tex`: ":12–13 "given the previous coherent
  control u_-" preamble; :70 "The regularization terms enter
  only~(stat-u)" now vacuous; :122 `Θ(ρ,t,u_-)` signature → `Θ(ρ,t)`;
  :171 `-\nu/w_delta` missing backslash; :172 bare-ASCII "u_sat = u_max
  sgn(u^(I))" in prose; :328 `W_r` declared but unused in eq:QP-multi.
- `Chapters/Applications.tex`: :427 w_r table row; :464 "five carried
  commands"; :480 "the worst carried command"; :547 "(w_r=0, w_u≡1)"; the
  0.55 / 0.45 / margin chain (§2 above).
- Dangling refs (5 undefined in main.log): `sec:regularization`
  (Applications :112, :177, :229) and `prob:funnel` (Applications :392).
- `Chapters/Prelimanaries.tex`: :274 typos `[0.\gamma_{max}]` →
  `[0,\gamma_{\max}]`, `u_{max}` → `u_{\max}`; :182 `(1-\theta_b),\epsilon(0)`
  stray comma.
- `main.tex:60`: `\weq` macro now used only by the stale Algo lines;
  `Chapters/Table.tex:1` comment still mentions it.

## 6. Mac checklist

1. `python sim_prototype/test/test_backends.py` — python-vs-numba
   trajectory equality (new guard; needs numba).
2. `python sim_prototype/test/test_controller.py` — closed form vs OSQP
   (needs osqp).
3. Re-render `fig_bell_chatter.pdf`:
   `cd sim_prototype && python run_bell.py` then
   `cd ../figsrc && python fig_signflip.py` (last-writer-wins on
   fig_chatter_robustness.pdf, as before). All other figures unchanged
   (<5e-7 trajectory equivalence).
4. Note: run_bell.py's *printed summary* still uses its own funnel
   (ε_T=0.20, T=1, κ=5 default) while the figure uses (ε_T=0.08, T=2,
   κ=50) — pre-existing inconsistency, left as-is deliberately; the
   figure and the paper's E2 numbers come from the figure config.
