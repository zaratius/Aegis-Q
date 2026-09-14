"""
Shared figure style for the QDBC set (IEEE TAC) -- built on SciencePlots.
Requires:  pip install SciencePlots      (import name: scienceplots)
"""
import matplotlib
matplotlib.use("pgf")

import matplotlib.pyplot as plt  
import scienceplots  

# ---- canvas widths (inches) ------------------------------------------------
COL = 3.40      # one IEEE column
FULL = 7.06     # two-column span (figure*)

# ---- semantic palette (Okabe-Ito) -----------------------------------------
OI = dict(black="#000000", orange="#ea801c", skyblue="#56B4E9",
          green="#198450", yellow="#FFEE00", blue="#00629b",
          vermillion="#FF7300", purple="#C300FF", grey="#999999", red="#FF0000")

C_FUNNEL   = "#000000"          # funnel boundary eps(t)               (solid)
C_SHELL    = OI["blue"]         # shell / buffer / shell-exit          (dashed)
C_BREACH   = OI["orange"]   # funnel-exit / breach / unregularized (solid)
C_CONFINED = OI["grey"]         # confined / nominal                   (dotted)
C_FEASIBLE = OI["green"]        # speed limit / feasible / regularized (solid)
GRID = "#E9E9E9"

LS_FUNNEL, LS_SHELL, LS_CONF = "-", (0, (4, 2)), (0, (1, 1.6))

# back-compat aliases
TEAL, ORANGE, RED, BLUE, GREY = C_FEASIBLE, C_BREACH, C_BREACH, C_SHELL, C_CONFINED
C_UNREG, C_REG, C_REF, C_FUN = C_BREACH, C_FEASIBLE, C_CONFINED, C_FUNNEL


def apply():
    """SciencePlots (science + ieee + grid), rendered via pgf+pdflatex."""
    plt.style.use(["science", "ieee", "grid"])
    plt.rcParams.update({
        # --- render through pgf so text is CM via pdflatex (no dvipng/gs) ---
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "text.usetex": False,            # pgf shells out to pdflatex itself
        # --- vector output, tight bbox ---
        "savefig.dpi": 600, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        # --- compact sizes for the page budget (SciencePlots ieee is ~8pt) ---
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        # --- keep the grid a hairline and horizontal-only (cleaner for bars) ---
        "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.4, "grid.alpha": 0.6,
        "legend.frameon": False,
    })


def funnel_curves(ax, t, eps, theta_b, fill=True):
    """Shared funnel vocabulary: boundary eps(t) (black solid) + shell
    (1-theta_b) eps(t) (blue dashed, relative buffer), with an optional faint
    buffer fill."""
    shell = (1.0 - theta_b) * eps
    if fill:
        ax.fill_between(t, shell, eps, color=C_SHELL, alpha=0.08, lw=0,
                        zorder=1)
    ax.plot(t, eps, color=C_FUNNEL, lw=1.2, ls=LS_FUNNEL, zorder=6)
    ax.plot(t, shell, color=C_SHELL, lw=0.9, ls=LS_SHELL, zorder=6)
