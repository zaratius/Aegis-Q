"""
One visual language for the QDBC figure set (IEEE TAC).

Semantic vocabulary, reused by every figure:
    funnel boundary eps(t)       black, solid        -- the specification
    shell / buffer / shell-exit  blue,  dashed       -- design-margin layer
    funnel-exit / breach         vermillion, solid   -- the safety event
    confined / nominal           grey,  dotted       -- uneventful paths
    speed limit / feasible / reg green, solid        -- feasibility / regularized

Okabe-Ito (colour-blind safe); every series also carries a line style/marker so
panels survive grayscale. Computer Modern via the pgf backend (matches the
IEEEtran body); two spines, hairline y-grid, direct labels. Sizing is tight for
the two-column page budget. If the paper moves to an ieeecolor/Times style,
switch the two font lines in _PGF_PREAMBLE to newtxtext/newtxmath.
"""
import matplotlib
matplotlib.use("pgf")

_PGF_PREAMBLE = "\n".join([
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amsmath,amssymb,mathtools}",
    r"\usepackage{bm}",
])

import matplotlib.pyplot as plt  # noqa: E402

# ---- canvas widths (inches) ------------------------------------------------
COL = 3.40      # one IEEE column
FULL = 7.06     # two-column span (figure*)

# ---- semantic palette (Okabe-Ito) -----------------------------------------
OI = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9",
          green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7", grey="#999999")

C_FUNNEL   = "#000000"          # funnel boundary eps(t)              (solid)
C_SHELL    = OI["blue"]         # shell / buffer / shell-exit         (dashed)
C_BREACH   = OI["vermillion"]   # funnel-exit / breach / unregularized(solid)
C_CONFINED = OI["grey"]         # confined / nominal                  (dotted)
C_FEASIBLE = OI["green"]        # speed limit / feasible / regularized(solid)
GRID = "#E9E9E9"

LS_FUNNEL, LS_SHELL, LS_CONF = "-", (0, (4, 2)), (0, (1, 1.6))

# back-compat aliases for any untouched scripts
TEAL, ORANGE, RED, BLUE, GREY = C_FEASIBLE, C_BREACH, C_BREACH, C_SHELL, C_CONFINED
C_UNREG, C_REG, C_REF, C_FUN = C_BREACH, C_FEASIBLE, C_CONFINED, C_FUNNEL


def apply():
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": _PGF_PREAMBLE,
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 1.0,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
    })


def funnel_curves(ax, t, eps, s_b, fill=True):
    """Shared funnel vocabulary: boundary eps(t) (black solid) + shell eps-s_b
    (blue dashed), with an optional faint buffer fill."""
    if fill:
        ax.fill_between(t, eps - s_b, eps, color=C_SHELL, alpha=0.08, lw=0,
                        zorder=1)
    ax.plot(t, eps, color=C_FUNNEL, lw=1.2, ls=LS_FUNNEL, zorder=6)
    ax.plot(t, eps - s_b, color=C_SHELL, lw=0.9, ls=LS_SHELL, zorder=6)
