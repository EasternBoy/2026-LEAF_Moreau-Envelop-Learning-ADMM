"""Box plots of optimality gap and constraint violation for the power-grid
benchmark, using the stored Julia and DC3 per-instance results.

Run from the repository root after benchmarking both horizons::

    python experiments/power_grid/boxplot.py              # both g_opt = 1% and 0.01%
    python experiments/power_grid/boxplot.py --gopt 1.0   # one target

Julia and DC3 results come from ``results/power_grid/table/``. Both quantities use logarithmic axes;
exact zeros are drawn at ``FLOOR``.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GAPS = (1.0, 0.01)   # table.jl writes results/power_grid/table/gap=1.0 and gap=0.01
TABLE_OUT = os.path.join(REPO, "results", "power_grid", "table")
DC3_OUT = TABLE_OUT
FIGURES_OUT = os.path.join(REPO, "results", "power_grid", "figures")
HORIZONS = [96, 192]
VIOL_TOL = 1e-4   # feasibility tolerance drawn on the constraint-violation plots
FLOOR = 1e-17
# Compensate for the larger canvas being reduced to fit the paper.
BASE_SIZE = 22
LABEL_SIZE = 32
TICK_SIZE = 1.1 * LABEL_SIZE
XTICK_SIZE = 0.95 * LABEL_SIZE   # method names must fit side by side in a 7-inch panel
METHODS = [
    ("ADMM", "#2a78d6"),
    ("sMEL-ADMM", "#eb6834"),
    ("DC3", "#1baf7a"),
]


def julia_results_file(N: int, g_opt: float) -> str:
    path = os.path.join(TABLE_OUT, f"gap={g_opt}", f"results_N={N}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Run experiments/power_grid/table.jl with N={N} and g_opt={g_opt}."
        )
    return path


def read_columns(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No result rows in {path}")
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def load_julia(method: str, N: int, g_opt: float) -> tuple[np.ndarray, np.ndarray]:
    data = read_columns(julia_results_file(N, g_opt))
    keep = data["solver"] == method
    if not np.any(keep):
        raise ValueError(f"No {method} rows in {julia_results_file(N, g_opt)}")
    gap = data["opt_gap_pct"][keep].astype(float)
    violation = data["feasibility_residual"][keep].astype(float)
    return gap, violation


def load_dc3(N: int) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(DC3_OUT, f"dc3_clarabel_results_N={N}.csv")
    data = read_columns(path)
    gap = data["opt_gap_pct"].astype(float)
    if "constraint_violation" in data:
        violation = data["constraint_violation"].astype(float)
    else:  # Backward compatibility with results generated before this column existed.
        violation = np.maximum.reduce([
            data["eq_max"].astype(float),
            data["ineq_max"].astype(float),
            data["domain_max"].astype(float),
            np.zeros(gap.size),
        ])
    return gap, violation


def load(method: str, N: int, g_opt: float) -> tuple[np.ndarray, np.ndarray]:
    return load_dc3(N) if method == "DC3" else load_julia(method, N, g_opt)


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
    ax.set_xticklabels([method for method, _ in METHODS], rotation=0, ha="center")


def figure(metric: int, ylabel: str, references, filename: str, g_opt: float):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(7 * len(HORIZONS), 6),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for panel, (N, ax) in enumerate(zip(HORIZONS, axes)):
        values = [np.maximum(load(method, N, g_opt)[metric], FLOOR) for method, _ in METHODS]
        boxes(ax, values)
        for y, label, linestyle, *where in references:
            ax.axhline(y, color="#52514e", linestyle=linestyle, linewidth=1)
            # Label above the line in the first panel, or (where = ["below"]) below it in the
            # last panel, whose upper-left area is free of boxes.
            above = where != ["below"]
            if panel == (0 if above else len(HORIZONS) - 1):
                ax.text(0.55, y * (1.6 if above else 0.6), label, color="#52514e",
                        fontsize=LABEL_SIZE, va="bottom" if above else "top")
        ax.set_title(f"$N = {N}$", fontsize=2 * BASE_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        ax.tick_params(axis="x", labelsize=XTICK_SIZE)
    axes[0].set_ylabel(ylabel, fontsize=XTICK_SIZE)   # same size as the method names
    path = os.path.join(FIGURES_OUT, f"{filename}_optgap={g_opt:g}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gopt", type=float, nargs="+", default=list(GAPS),
                        help="gap targets in %% (default: 1.0 0.01)")
    args = parser.parse_args()
    os.makedirs(FIGURES_OUT, exist_ok=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    for g_opt in args.gopt:
        figure(0, r"Optimality gap (\%)",
               [(g_opt, rf"{{\boldmath$g_{{\mathrm{{opt}}}} = {g_opt:g}\%$}}", "--")],
               "power_grid_gap_boxplot", g_opt)
        figure(1, "Constraint violation",
               [(VIOL_TOL, r"{\boldmath$c_v = 10^{-4}$}", "--", "below")],
               "power_grid_viol_boxplot", g_opt)


if __name__ == "__main__":
    main()
