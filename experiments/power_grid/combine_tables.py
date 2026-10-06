"""Combine the power-grid benchmark tables into one table per horizon N.

Run from the repository root, after experiments/power_grid/table.jl (--gopt=1.0 and
--gopt=0.01) and experiments/power_grid/table_benchmark.py::

    python3 experiments/power_grid/combine_tables.py

Reads results/power_grid/table/gap=<g>/summary_N=<N>*.csv and
results/power_grid/table/dc3_clarabel_summary_N=<N>.json, and writes
results/power_grid/table_N=<N>.md and .tex with, for each gap target, the solving time,
optimality gap and constraint violation of every solver.

DC3 + correction has no gap-based stopping rule, so its one run is reported under both
targets; its solving time is "Unable" when its maximum gap exceeds the target.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "power_grid"
TABLE_DIR = RESULTS / "table"
GAPS = (1.0, 0.01)
HORIZONS = (96, 192)
ORDER = ("ADMM", "IPOPT", "MadNLP", "DC3 + correction", "MEL-ADMM", "sMEL-ADMM")
DC3 = "DC3 + correction"


def gap_label(g: float) -> str:
    return f"{g:g}"


def load_summary(N: int, g: float) -> tuple[dict, int]:
    files = sorted((TABLE_DIR / f"gap={g}").glob(f"summary_N={N}*.csv"))   # table.jl writes gap=1.0, gap=0.01
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one summary for N={N}, gap={g}%, found {[f.name for f in files]}")
    rows = {r["solver"]: r for r in csv.DictReader(files[0].open())}
    samples = {int(r["samples"]) for r in rows.values()}
    if len(samples) != 1:
        raise ValueError(f"Mixed sample counts in {files[0]}: {samples}")
    return {name: {"time_mean": float(r["time_mean_ms"]), "time_max": float(r["time_max_ms"]),
                   "gap_mean": float(r["opt_gap_mean_pct"]), "gap_max": float(r["opt_gap_max_pct"]),
                   "viol_mean": float(r["constr_viol_mean"]), "viol_max": float(r["constr_viol_max"])}
            for name, r in rows.items()}, samples.pop()


def load_dc3(N: int) -> tuple[dict, int] | None:
    path = TABLE_DIR / f"dc3_clarabel_summary_N={N}.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    return {"time_mean": s["solve_time_mean_ms"], "time_max": s["solve_time_max_ms"],
            "gap_mean": s["opt_gap_mean_pct"], "gap_max": s["opt_gap_max_pct"],
            "viol_mean": s["constraint_violation_mean"], "viol_max": s["constraint_violation_max"]}, int(s["samples"])


def sci(x: float) -> str:
    return "0" if x == 0 else f"{x:.1e}"


def cells(row: dict, unable: bool) -> tuple[str, str, str]:
    time = "Unable" if unable else f"{row['time_mean']:.2f} ({row['time_max']:.2f})"
    return (time, f"{sci(row['gap_mean'])} ({sci(row['gap_max'])})",
            f"{sci(row['viol_mean'])} ({sci(row['viol_max'])})")


def build(N: int) -> tuple[str, str]:
    per_gap, samples = {}, set()
    for g in GAPS:
        per_gap[g], n = load_summary(N, g)
        samples.add(n)
    dc3 = load_dc3(N)
    if dc3 is not None:
        for g in GAPS:
            per_gap[g][DC3] = dc3[0]
        samples.add(dc3[1])
    if len(samples) != 1:
        raise ValueError(f"N={N}: the tables use different sample counts {sorted(samples)}")
    n = samples.pop()
    solvers = [s for s in ORDER if all(s in per_gap[g] for g in GAPS)]

    rows = {}
    for s in solvers:
        rows[s] = []
        for g in GAPS:
            r = per_gap[g][s]
            rows[s].extend(cells(r, unable=(s == DC3 and r["gap_max"] > g)))
    fastest = {g: min((per_gap[g][s]["time_mean"], s) for s in solvers
                      if not (s == DC3 and per_gap[g][s]["gap_max"] > g))[1] for g in GAPS}

    head = " | ".join(f"g_opt ≤ {gap_label(g)}%: {k}" for g in GAPS for k in ("time", "gap", "viol."))
    md = [f"# Power-grid benchmark, N = {N} ({3 * N} scalar variables), CPU, {n} instances", "",
          "All cells are mean (max). Time: solving time in ms; gap: optimality gap in %; viol.: largest "
          "violation of the original constraints. **Bold**: lowest mean solving time for that target. "
          "DC3 + correction has no gap-based stop: one run, shown under both targets, "
          "\"Unable\" when its maximum gap exceeds the target.", "",
          f"| Solver | {head} |", "|---|" + "---:|" * (3 * len(GAPS))]
    for s in solvers:
        c = list(rows[s])
        for i, g in enumerate(GAPS):
            if fastest[g] == s:
                c[3 * i] = f"**{c[3 * i]}**"
        md.append(f"| {s} | " + " | ".join(c) + " |")

    tex = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{Computation time (ms) benchmark at $N = {N}$ with CPU computation over {n} samples}}",
           rf"\label{{tab:power_grid_N{N}}}", r"\begin{tabular}{l" + "ccc" * len(GAPS) + "}", r"\hline",
           " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$g_{{opt}} \leq {gap_label(g)}\%$}}" for g in GAPS) + r" \\",
           " & " + " & ".join(["solving time", "optimality gap (\\%)", "constr. viol."] * len(GAPS)) + r" \\",
           " & " + " & ".join(["mean (max)"] * (3 * len(GAPS))) + r" \\", r"\hline"]
    for s in solvers:
        c = list(rows[s])
        for i, g in enumerate(GAPS):
            if fastest[g] == s:
                c[3 * i] = rf"\textbf{{{c[3 * i]}}}"
        tex.append(f"{s} & " + " & ".join(c) + r" \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def main() -> None:
    for N in HORIZONS:
        md, tex = build(N)
        for ext, text in (("md", md), ("tex", tex)):
            path = RESULTS / f"table_N={N}.{ext}"
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO_ROOT.parent)}")


if __name__ == "__main__":
    main()
