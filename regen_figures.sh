#!/usr/bin/env bash
#
# Regenerate all SCCQS paper figures (and their data).  Self-contained in
# the Aegis repo: everything lands in $AEGIS/figures/ -- no paper repo is
# needed on this machine.  Transfer the PDFs to the SCCQS clone afterwards
# (git commit/push from here, or copy figures/*.pdf directly).
#
#   Usage:  bash regen_figures.sh [--sweeps]
#
#     --sweeps    also run/extend the eta and dt sweeps that feed
#                 fig_breach_refinement.  These are ACCUMULATIVE and already
#                 complete (300 seeds/point), so this is normally a no-op --
#                 only needed if eta_sweep_results.json is lost or extended.
#
# Runs on the Mac in the "aegisq" conda environment.
set -euo pipefail

# ---------------------------------------------------------------- config ----
# The repo root is wherever this script lives -- no path to adjust.
AEGIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="aegisq"
ROBUST_M=1000                      # ensemble size for the robustness study
                                   # (script default is 40 -- the paper uses 1000)

DO_SWEEPS=0
for arg in "$@"; do
  case "$arg" in
    --sweeps)  DO_SWEEPS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

LOG="$AEGIS/regen_$(date +%Y%m%d_%H%M%S).log"
START=$(date +%s)

# ------------------------------------------------------------ conda env ----
# `conda activate` needs the shell hook in a non-interactive script.
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH." >&2; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "env: $CONDA_ENV  ->  $(python -c 'import sys; print(sys.executable)')"

# ------------------------------------------------------------- preflight ----
cd "$AEGIS"
[ -d figsrc ] && [ -d sim_prototype ] || { echo "ERROR: \$AEGIS=$AEGIS is not the Aegis repo." >&2; exit 1; }

python - <<'PY'
import importlib.util, sys
missing = [m for m in ("numpy", "scipy", "matplotlib", "numba") if not importlib.util.find_spec(m)]
if missing:
    sys.exit("ERROR: missing packages in this env: " + ", ".join(missing))
print("deps ok: numpy, scipy, matplotlib, numba")
PY

if [ ! -f sim_prototype/eta_sweep_results.json ] && [ "$DO_SWEEPS" -eq 0 ]; then
  echo "ERROR: sim_prototype/eta_sweep_results.json missing and --sweeps not given." >&2
  echo "       fig_breach_refinement cannot be built without it.  Re-run with --sweeps." >&2
  exit 1
fi

mkdir -p figures
echo "logging to $LOG"
echo

# ------------------------------------------------------------- pipeline ----
# Everything below tees to the log so the closing summary can grep the
# reported statistics back out and compare them with the paper.
{
  if [ "$DO_SWEEPS" -eq 1 ]; then
    echo "### [0/6] eta + dt sweeps (accumulative; no-op if already complete)"
    ( cd sim_prototype
      python run_eta_sweep.py eta 300 0
      python run_eta_sweep.py dt  300 0 )
    echo
  fi

  echo "### [1/6] fig_feas.py       -> fig_feasibility.pdf, fig_feasibility_small_buffer.pdf"
  ( cd figsrc && python fig_feas.py )        # also writes figsrc/feas_data.npz
  echo

  echo "### [2/6] fig_dtref.py      -> fig_breach_refinement.pdf   (plots the sweep JSON)"
  ( cd figsrc && python fig_dtref.py )
  echo

  echo "### [3/6] run_e1_acc.py     -> fig_qubit_confinement.pdf   (M=1000 Monte Carlo)"
  ( cd sim_prototype && python run_e1_acc.py )
  echo

  echo "### [4/6] run_bell.py       -> fig_bell_chatter.pdf"
  echo "###       (also writes fig_chatter_robustness.pdf -- overwritten in step 5)"
  ( cd sim_prototype && python run_bell.py )
  echo

  echo "### [5/6] fig_signflip.py   -> fig_chatter_robustness.pdf  (YOUR chosen version)"
  # MUST run after run_bell.py: both write fig_chatter_robustness.pdf and the
  # last writer wins.  See the note at the end of this script.
  ( cd figsrc && python fig_signflip.py )
  echo

  echo "### [6/6] run_robustness_acc.py $ROBUST_M -> fig_robustness.pdf  (M=$ROBUST_M per design)"
  ( cd sim_prototype && python run_robustness_acc.py "$ROBUST_M" )
  echo
} 2>&1 | tee "$LOG"

# -------------------------------------------------------------- verify ----
echo
echo "=== verifying the 7 figures the paper includes ==="
WANT=(fig_breach_refinement.pdf
      fig_qubit_confinement.pdf
      fig_bell_chatter.pdf
      fig_chatter_robustness.pdf
      fig_feasibility.pdf
      fig_feasibility_small_buffer.pdf
      fig_robustness.pdf)

FAIL=0
for f in "${WANT[@]}"; do
  p="figures/$f"
  if [ ! -f "$p" ]; then
    printf '  MISSING  %s\n' "$f"; FAIL=1
  elif [ "$(stat -f %m "$p")" -lt "$START" ]; then
    printf '  STALE    %s  (not rewritten by this run)\n' "$f"; FAIL=1
  else
    printf '  ok       %-38s %6s KB\n' "$f" "$(( $(stat -f %z "$p") / 1024 ))"
  fi
done
[ "$FAIL" -eq 0 ] || { echo; echo "ERROR: figure generation incomplete -- not syncing." >&2; exit 1; }

# ------------------------------------------------- numbers cross-check ----
# The runs are deterministically seeded, so these should reproduce the values
# printed in the paper.  If any differ, the paper text needs updating.
echo
echo "=== reported statistics (compare against the paper) ==="
grep -iE "confin|shell|funnel-exit|excursion|[0-9]+/[0-9]+|CI|%" "$LOG" \
  | grep -viE "^###|writing|wrote|integrated|workers" | sed 's/^/  /' || true
cat <<'EOF'

  Paper values for reference:
    qubit confinement (M=1000)     ~92.7%   [90.9, 94.2]
    shell-exit  @ eta=0.6, M=300    25.3%   [20.5, 30.7]
    funnel-exit @ eta=0.6, M=300     6.0%   [ 3.6,  9.3]
    robust      confinement        931/1000 = 93.1%  [91.3, 94.6]
    optimistic  confinement        784/1000 = 78.4%  [75.7, 80.9]
    worst excursion  xi/eps         1.93 (robust)  vs  3.03 (optimistic)
EOF

# ------------------------------------------------------------- handoff ----
echo
echo "=== all figures are in $AEGIS/figures/ ==="
echo "The paper repo (SCCQS) lives on the Windows machine.  To hand off:"
echo "  git -C \"$AEGIS\" add figures && git -C \"$AEGIS\" commit -m 'regen figures' && git -C \"$AEGIS\" push"
echo "then on the Windows side: pull Aegis, copy Aegis/figures/*.pdf into"
echo "SCCQS/figures/, and recompile:"
echo "  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"

echo
printf 'total wall time: %d min %d s\n' $(( ($(date +%s)-START)/60 )) $(( ($(date +%s)-START)%60 ))
