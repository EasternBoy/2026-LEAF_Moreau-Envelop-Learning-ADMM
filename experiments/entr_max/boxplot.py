"""Box plots of optimality gap and constraint violation for the maximum-entropy
cone program: IPOPT, sLME-ADMM and DC3 + correction on every (n, m) of
table.py, from the stored data in results/entr_max/table/.

    python experiments/entr_max/boxplot.py

IPOPT and sLME-ADMM use the g_opt <= 0.1% runs (the same ones as the table's
Opt. gap rows); DC3 has a single run.  Both quantities span many orders of
magnitude, so the y-axes are logarithmic and exact zeros are drawn at FLOOR.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "entr_max", "table")
FIGURES_OUT = os.path.join(REPO, "results", "entr_max", "figures")
SIZES = [(100, 10), (1000, 10), (1000, 100)]
GNAME = "0.1"
FEAS_TOL = 1e-4
FLOOR = 1e-17                                    # where exact zeros are drawn on the log axis
BASE_SIZE = 10
LABEL_SIZE = 15                                  # y-label and reference-line labels: 1.5x the base font size
TICK_SIZE = 1.1 * LABEL_SIZE
YLABEL_SIZE = 1.5 * LABEL_SIZE
METHODS = [("IPOPT", "#2a78d6"), ("sLME-ADMM", "#eb6834"), ("DC3", "#1baf7a")]


def load(meth, n, m):
    name = f"DC3-n={n}-m={m}.npz" if meth.startswith("DC3") else f"{meth}-gopt={GNAME}-n={n}-m={m}.npz"
    d = np.load(os.path.join(OUT, name))
    return d["gap_pct"], d["max_viol"]


def boxes(ax, vals):
    bp = ax.boxplot(vals, widths=0.55, patch_artist=True, whis=(0, 100),
                    medianprops={"color": "#0b0b0b", "linewidth": 3})
    for box, (_, c) in zip(bp["boxes"], METHODS):
        box.set(facecolor=c, edgecolor=c, alpha=0.85, linewidth=2)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color="#52514e", linewidth=2)
    ax.set_yscale("log")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels([meth for meth, _ in METHODS])


def figure(k, ylabel, refs, fname):
    """One figure, one panel per (n, m); k = 0 gap, 1 violation; refs = [(y, label, linestyle)]."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4 * len(SIZES), 4), sharey=True, constrained_layout=True)
    for j, ((n, m), ax) in enumerate(zip(SIZES, axes)):
        boxes(ax, [np.maximum(load(meth, n, m)[k], FLOOR) for meth, _ in METHODS])
        for y, lab, ls in refs:
            ax.axhline(y, color="#52514e", linestyle=ls, linewidth=1)
            if j == 0:
                ax.text(0.55, y * 1.6, lab, color="#52514e", fontsize=LABEL_SIZE)
        ax.set_title(f"$n = {n},\\ m = {m}$", fontsize=2 * BASE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
    axes[0].set_ylabel(ylabel, fontsize=YLABEL_SIZE)
    os.makedirs(FIGURES_OUT, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_OUT, f"{fname}.pdf"))
    print(f"saved {os.path.join(FIGURES_OUT, fname)}.pdf")


def main():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": BASE_SIZE, "axes.spines.top": False, "axes.spines.right": False,
                         "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}"})
    figure(0, r"Optimality gap (\%)", [(float(GNAME), rf"$\bm{{g_{{\mathrm{{\bf opt}}}} = {GNAME}\%}}$", "--")],
           "entr_max_gap_boxplot")
    figure(1, "Constraint violation", [(FEAS_TOL, r"$\bm{c_v = 10^{-4}}$", "--")], "entr_max_viol_boxplot")


if __name__ == "__main__":
    main()
