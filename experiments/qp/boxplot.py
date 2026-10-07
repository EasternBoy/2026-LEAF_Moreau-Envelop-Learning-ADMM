"""QP gap/violation box plots, following experiments/entr_max/boxplot.py.

    python3 experiments/qp/boxplot.py

Read sLME-ADMM and DC3 batch-1 results saved by table.jl/table.py.
Logarithmic axes display exact zeros at FLOOR.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "qp", "table")
SUFFIX = "n=100-neq=50-m=50-samples=833"
GNAME = "1.0"
FEAS_TOL = 1e-4
FLOOR = 1e-17
BASE_SIZE = 10
LABEL_SIZE = 15
TICK_SIZE = 1.1 * LABEL_SIZE
YLABEL_SIZE = 1.5 * LABEL_SIZE
METHODS = [("sLME-ADMM", "#eb6834"), ("DC3", "#1baf7a")]
FILES = {"sLME-ADMM": f"sLME-ADMM-{SUFFIX}-tol=0.01-gopt={GNAME}.npz",
         "DC3": f"DC3-{SUFFIX}.npz"}


def load(method):
    with np.load(os.path.join(OUT, FILES[method])) as saved:
        return saved["gap_pct"], saved["max_viol"]


def boxes(ax, values):
    bp = ax.boxplot(values, widths=0.55, patch_artist=True, whis=(0, 100),
                    medianprops={"color": "#0b0b0b", "linewidth": 3})
    for box, (_, color) in zip(bp["boxes"], METHODS):
        box.set(facecolor=color, edgecolor=color, alpha=0.85, linewidth=2)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color="#52514e", linewidth=2)
    ax.set_yscale("log")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels([method for method, _ in METHODS])


def figure(k, ylabel, refs, filename):
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    boxes(ax, [np.maximum(load(method)[k], FLOOR) for method, _ in METHODS])
    if k == 0:
        ax.set_ylim(top=max(10.0, ax.get_ylim()[1]))
    for y, label, style in refs:
        ax.axhline(y, color="#52514e", linestyle=style, linewidth=1)
        if k == 1:
            ax.text(0.55, y / 1.6, label, color="#52514e", fontsize=LABEL_SIZE, va="top")
        else:
            ax.text(0.55, y * 1.6, label, color="#52514e", fontsize=LABEL_SIZE)
    ax.set_title(r"$n = 100,\ n_{\mathrm{eq}} = 50,\ n_{\mathrm{ineq}} = 50$", fontsize=2 * BASE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=YLABEL_SIZE)
    path = os.path.join(OUT, f"{filename}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": BASE_SIZE,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure(0, r"Optimality gap (\%)", [(float(GNAME), rf"$g_{{\mathrm{{opt}}}} = {GNAME}\%$", "--")],
           "qp_gap_boxplot")
    figure(1, "Constraint violation", [(FEAS_TOL, r"$c_v = 10^{-4}$", "--")], "qp_viol_boxplot")


if __name__ == "__main__":
    main()
