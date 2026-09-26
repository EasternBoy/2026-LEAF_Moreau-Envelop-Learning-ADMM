"""Box plots of optimality gap and constraint violation for the power-grid
benchmark, using the stored Julia and DC3 per-instance results.

Run from the repository root after benchmarking both horizons::

    python script/power_grid/power_grid_boxplot.py

Julia results come from ``power_grid_table.jl`` in
``data/solving_data/power_table_time_gap=0.01/``. DC3 results come from
``DC3/power_grid/benchmark/results/``. Both quantities use logarithmic axes;
exact zeros are drawn at ``FLOOR``.
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G_OPT = 0.01
OUT = os.path.join(REPO, "data", "solving_data", f"power_table_time_gap={G_OPT}")
DC3_OUT = os.path.join(REPO, "DC3", "power_grid", "benchmark", "results")
HORIZONS = [96, 192]
FLOOR = 1e-17
BASE_SIZE = 10
LABEL_SIZE = 15
TICK_SIZE = 1.1 * LABEL_SIZE
YLABEL_SIZE = 1.5 * LABEL_SIZE
METHODS = [
    ("IPOPT", "#e377c2"),
    ("ADMM", "#2a78d6"),
    ("sMEL-ADMM", "#eb6834"),
    ("DC3", "#1baf7a"),
]


def julia_results_file(N: int) -> str:
    pattern = os.path.join(OUT, f"results_N={N}_*optgap={G_OPT}_*.csv")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Julia result matching {pattern}, found {len(matches)}. "
            f"Run power_grid_table.jl with N={N} and BENCHMARK_MODE=:time."
        )
    return matches[0]


def read_columns(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No result rows in {path}")
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def load_julia(method: str, N: int) -> tuple[np.ndarray, np.ndarray]:
    data = read_columns(julia_results_file(N))
    keep = data["solver"] == method
    if not np.any(keep):
        raise ValueError(f"No {method} rows in {julia_results_file(N)}")
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


def load(method: str, N: int) -> tuple[np.ndarray, np.ndarray]:
    return load_dc3(N) if method == "DC3" else load_julia(method, N)


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


def figure(metric: int, ylabel: str, references, filename: str):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(7 * len(HORIZONS), 5),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for panel, (N, ax) in enumerate(zip(HORIZONS, axes)):
        values = [np.maximum(load(method, N)[metric], FLOOR) for method, _ in METHODS]
        boxes(ax, values)
        for y, label, linestyle in references:
            ax.axhline(y, color="#52514e", linestyle=linestyle, linewidth=1)
            if panel == 0:
                ax.text(0.55, y * 1.6, label, color="#52514e", fontsize=LABEL_SIZE)
        ax.set_title(f"$N = {N}$", fontsize=2 * BASE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
    axes[0].set_ylabel(ylabel, fontsize=YLABEL_SIZE)
    path = os.path.join(OUT, f"{filename}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure(0, r"Optimality gap (\%)",
           [(G_OPT, rf"$g_{{\mathrm{{opt}}}} = {G_OPT}\%$", "--")],
           "power_grid_gap_boxplot")
    figure(1, "Constraint violation",
           [],
           "power_grid_viol_boxplot")


if __name__ == "__main__":
    main()
