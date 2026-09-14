#!/usr/bin/env bash
#
# Regenerate all SCCQS paper figures (and their data) for the FINAL
# certified designs (2026-08-09):
#
#   qubit: theta_b=0.20, eps_T=1e-4, gamma_max=3.90  -> certified 0.2392
#   Bell : theta_b=0.20, eps_T=1e-4, kappa=15 (demonstrated rate),
#          gamma_max=1.30, T=4                        -> certified 0.5679
#
# Self-contained in the Aegis repo: everything lands in $AEGIS/figures/.
# Transfer the PDFs to the SCCQS clone afterwards (copy figures/*.pdf).
#
#   Usage:  bash regen_figures.sh [--sweeps]
#
#     --sweeps    also run/extend the eta + dt sweeps (feed
#                 fig_breach_refinement) and the Bell dt-refinement
#                 points.  These are ACCUMULATIVE and already complete
#                 (300 seeds/point at the final design), so this is
#                 normally a no-op -- only needed if the JSON files are
#                 lost or being extended.
#
# Runs in the "aegisq" conda environment, on macOS or on Windows
# (Git Bash; falls back to the miniconda env python directly if the
# conda hook is unavailable).
#
# RETIRED from the paper pipeline (scripts remain in the repo for the
# record, but are no longer run here):
#   run_bell.py / fig_bell_chatter.pdf      -- chattering study replaced
#   fig_signflip.py / fig_chatter_robustness.pdf -- Table III retired
#     (the Bell instance is now verified by run_bell_mc.py: certified
#      envelope + M=1000 ensemble -> fig_bell_confinement.pdf)
set -euo pipefail

# ---------------------------------------------------------------- config ----
AEGIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="aegisq"
ROBUST_M=1000                      # ensemble size for the robustness study
BELL_M=1000                        # ensemble size for the certified Bell MC

DO_SWEEPS=0
for arg in "$@"; do
  case "$arg" in
    --sweeps)  DO_SWEEPS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

LOG="$AEGIS/regen_$(date +%Y%m%d_%H%M%S).log"
START=$(date +%s)

# ---------------------------------------------------- portable stat helpers ----
# GNU stat (Linux/Git-Bash) uses -c; BSD stat (macOS) uses -f.
mtime_of() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }
size_of()  { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

# ------------------------------------------------------------ python env ----
# Prefer the conda hook; fall back to the env's python directly (Windows
# Git Bash has no conda on PATH by default).
PY=""
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  PY="python"
else
  # Git Bash on Windows: HOME may not be the user profile, so try both.
  for cand in "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" \
              "$HOME/miniconda3/envs/$CONDA_ENV/python.exe" \
              "${USERPROFILE:-}/miniconda3/envs/$CONDA_ENV/python.exe"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { PY="$cand"; break; }
  done
fi
[ -n "$PY" ] || { echo "ERROR: no conda and no $CONDA_ENV env python found." >&2; exit 1; }
echo "env: $CONDA_ENV  ->  $("$PY" -c 'import sys; print(sys.executable)')"

# ------------------------------------------------------------- preflight ----
cd "$AEGIS"
[ -d figsrc ] && [ -d sim_prototype ] || { echo "ERROR: \$AEGIS=$AEGIS is not the Aegis repo." >&2; exit 1; }

"$PY" - <<'PYEOF'
import importlib.util, sys
missing = [m for m in ("numpy", "scipy", "matplotlib", "numba", "scienceplots")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit("ERROR: missing packages in this env: " + ", ".join(missing))
print("deps ok: numpy, scipy, matplotlib, numba, scienceplots")
PYEOF

# fail fast on unwritable directories (the pipeline writes JSON/npz/PDFs
# into all three; a permission problem should not surface minutes in)
mkdir -p figures
for d in sim_prototype figsrc figures; do
  t="$d/.write_test.$$"
  if ! ( : > "$t" ) 2>/dev/null; then
    echo "ERROR: directory '$d' is not writable by $(id -un)." >&2
    echo "       fix permissions (chmod u+w / chown / chflags nouchg) and re-run." >&2
    exit 1
  fi
  rm -f "$t"
done

if [ ! -f sim_prototype/eta_sweep_results.json ] && [ "$DO_SWEEPS" -eq 0 ]; then
  echo "ERROR: sim_prototype/eta_sweep_results.json missing and --sweeps not given." >&2
  echo "       fig_breach_refinement cannot be built without it.  Re-run with --sweeps." >&2
  exit 1
fi

echo "logging to $LOG"
echo

# ------------------------------------------------------------- pipeline ----
# Everything below tees to the log so the closing summary can grep the
# reported statistics back out and compare them with the memos/paper.
{
  if [ "$DO_SWEEPS" -eq 1 ]; then
    echo "### [0/6] sweeps (accumulative; no-op if already complete)"
    ( cd sim_prototype
      "$PY" run_eta_sweep.py eta 300 0
      "$PY" run_eta_sweep.py dt  300 0
      "$PY" run_bell_mc.py refine 300 )
    echo
  fi

  echo "### [1/6] fig_feas.py       -> fig_feasibility.pdf, fig_feasibility_small_buffer.pdf"
  ( cd figsrc && "$PY" fig_feas.py )         # also writes figsrc/feas_data.npz
  echo

  echo "### [2/6] fig_dtref.py      -> fig_breach_refinement.pdf   (plots the sweep JSON)"
  ( cd figsrc && "$PY" fig_dtref.py )
  echo

  echo "### [3/6] run_e1_acc.py     -> fig_qubit_confinement.pdf"
  echo "###       (M=1000 Monte Carlo + coarse-grid certified bound annotation)"
  ( cd sim_prototype && "$PY" run_e1_acc.py )
  echo

  echo "### [4/6] run_bell_mc.py envelope  -> certified Bell bound (refinement ladder)"
  # MUST run before step 5: the main run reads the certified bound out of
  # bell_mc_results.json for the figure annotation.  The ladder runs on the
  # numba prange kernel (all cores; ~2 min total, finest rung dominates);
  # it falls back to the serial NumPy path if numba is unavailable (~8 min).
  ( cd sim_prototype && "$PY" run_bell_mc.py envelope )
  echo

  echo "### [5/6] run_bell_mc.py $BELL_M    -> fig_bell_confinement.pdf  (M=$BELL_M ensemble)"
  ( cd sim_prototype && "$PY" run_bell_mc.py "$BELL_M" )
  echo

  echo "### [6/6] run_robustness_acc.py $ROBUST_M -> fig_robustness.pdf  (M=$ROBUST_M per design)"
  ( cd sim_prototype && "$PY" run_robustness_acc.py "$ROBUST_M" )
  echo
} 2>&1 | tee "$LOG"

# -------------------------------------------------------------- verify ----
echo
echo "=== verifying the 6 figures the paper includes ==="
WANT=(fig_breach_refinement.pdf
      fig_qubit_confinement.pdf
      fig_bell_confinement.pdf
      fig_feasibility.pdf
      fig_feasibility_small_buffer.pdf
      fig_robustness.pdf)

FAIL=0
for f in "${WANT[@]}"; do
  p="figures/$f"
  if [ ! -f "$p" ]; then
    printf '  MISSING  %s\n' "$f"; FAIL=1
  elif [ "$(mtime_of "$p")" -lt "$START" ]; then
    printf '  STALE    %s  (not rewritten by this run)\n' "$f"; FAIL=1
  else
    printf '  ok       %-38s %6s KB\n' "$f" "$(( $(size_of "$p") / 1024 ))"
  fi
done
[ "$FAIL" -eq 0 ] || { echo; echo "ERROR: figure generation incomplete -- not syncing." >&2; exit 1; }

# ------------------------------------------------- numbers cross-check ----
# The runs are deterministically seeded, so these should reproduce the
# values below (recorded in QUBIT_REDESIGN.md / BELL_MC_RESULTS.md §0).
# NOTE: the PAPER prose still carries pre-redesign numbers until the
# Section VI rewrite lands -- compare against the memos, not the old text.
echo
echo "=== reported statistics (compare against the memos) ==="
grep -iE "confin|shell|funnel-exit|excursion|certified|bound|[0-9]+/[0-9]+|CI|%" "$LOG" \
  | grep -viE "^###|writing|wrote|integrated|workers" | sed 's/^/  /' || true
cat <<'EOF'

  Final-design reference values (memos, 2026-08-09; seeds 5000+k / 3000+k):
    qubit certified bound              P <= 0.2392   (Doob floor 0.2344)
    qubit confinement (M=1000)          96.1%   [94.7, 97.2]
    qubit shell-exit / funnel-exit      13.8%   [11.7, 16.1]  /  3.9% [2.8, 5.3]
    eta sweep @ 0.6 (M=300, t<=0.5)     10.7%   [ 7.4, 14.7]  /  5.3% [3.1, 8.5]
    Bell certified bound               P <= 0.5679   (Doob floor 0.5602)
    Bell confinement (M=1000)           99.5%   [98.8, 99.8]
    Bell shell-exit / funnel-exit        1.0%   [ 0.5,  1.8]  /  0.5% [0.2, 1.2]
    robust      confinement (kappa=3)  961/1000 = 96.1%   worst xi/eps 1.51
    optimistic  confinement (kappa=3)  784/1000 = 78.4%   worst xi/eps 4.02
EOF

# ------------------------------------------------------------- handoff ----
echo
echo "=== all figures are in $AEGIS/figures/ ==="
SCCQS_GUESS="$(cd "$AEGIS/.." && pwd)/SCCQS"
if [ -d "$SCCQS_GUESS/figures" ]; then
  echo "Paper repo detected at $SCCQS_GUESS -- to hand off, copy:"
  for f in "${WANT[@]}"; do
    echo "  cp \"$AEGIS/figures/$f\" \"$SCCQS_GUESS/figures/\""
  done
else
  echo "Copy the six PDFs above into the SCCQS clone's figures/ directory, then"
fi
echo "recompile the paper:"
echo "  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"

echo
printf 'total wall time: %d min %d s\n' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 ))
