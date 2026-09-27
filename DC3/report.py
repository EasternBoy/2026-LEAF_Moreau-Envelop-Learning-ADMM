"""Turn ``benchmark.json`` (+ optional Julia baseline results) into tables and plots.

    python -m DC3.report --app power_grid

Baselines are read from ``DC3/results/<app>[-tag]/julia_baselines.json``, which is
written by the scripts in ``DC3/julia``.  If that file is missing the report says
so explicitly instead of inventing numbers.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .common.io_utils import DC3_ROOT, load_json, save_csv
from .common.metrics import format_table

METHOD_NOTES = {
    "Ipopt": "reference NLP solver used by experiments/*/benchmark*.jl",
    "Ipopt(early-stop)": "Ipopt with the repo's optimality-gap callback (benchmark setting)",
    "Clarabel(cvxpy)": "conic reference solver, tol 1e-9",
    "sLME-ADMM": "problems/entr_max/lme_admm.jl :: sLME_ADMM",
    "LME-ADMM(split)": "problems/power_grid/lme_admm.jl :: LME_ADMM_split",
    "DC3 + correction": "this implementation (completion + correction)",
}


def _get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return default if d is None else d


def collect_rows(bench: dict, julia: dict | None) -> list[dict]:
    if bench.get("schema_version", 0) < 2:
        raise ValueError("Legacy benchmark: rerun benchmark with domain-valid metrics and matched timing before reporting")
    if julia is not None and julia.get("schema_version", 0) < 2:
        raise ValueError("Legacy Julia baselines: rerun the Julia driver before reporting")
    rows = []
    ref = bench.get("reference")
    if ref:
        rows.append({
            "method": f"{ref['solver']}(cvxpy)",
            "obj_mean": _get(ref, "feasibility", "obj_mean"),
            "gap_mean": _get(ref, "feasibility", "gap_pct_mean"),
            "gap_max": _get(ref, "feasibility", "gap_pct_max"),
            "feas_rate": _get(ref, "feasibility", "feasible_rate"),
            "domain_rate": _get(ref, "feasibility", "domain_valid_rate"),
            "eq_max": _get(ref, "feasibility", "eq_max_max"),
            "ineq_max": _get(ref, "feasibility", "ineq_max_max"),
            "lat_median_ms": _get(ref, "latency_ms", "median_ms"),
            "note": "reference (gap defined as 0)",
        })
    for key, label in (("corrected", "DC3 + correction"),):
        a = bench["dc3"][key]
        rows.append({
            "method": label,
            "obj_mean": a.get("obj_mean"),
            "gap_mean": a.get("gap_pct_mean"),
            "gap_max": a.get("gap_pct_max"),
            "gap_mean_feasible": a.get("gap_pct_feasible_mean"),
            "feas_rate": a.get("feasible_rate"),
            "domain_rate": a.get("domain_valid_rate"),
            "eq_max": a.get("eq_max_max"),
            "ineq_max": a.get("ineq_max_max"),
            "domain_max": a.get("domain_max_max"),
            "lat_median_ms": _get(bench, "dc3", "latency", "single_instance", "median_ms")
            if key == "corrected" else _get(bench, "dc3", "latency", "stage_predict_b1", "median_ms"),
            "note": METHOD_NOTES.get(label, ""),
        })
    if julia:
        for name, r in julia.get("methods", {}).items():
            rows.append({
                "method": name + (" [oracle]" if r.get("oracle_assisted") else ""),
                "obj_mean": r.get("obj_mean"),
                "gap_mean": r.get("gap_pct_mean"),
                "gap_max": r.get("gap_pct_max"),
                "gap_mean_feasible": r.get("gap_pct_feasible_mean"),
                "domain_rate": r.get("domain_valid_rate"),
                "feas_rate": r.get("feasible_rate"),
                "eq_max": r.get("eq_max"),
                "ineq_max": r.get("ineq_max"),
                "lat_median_ms": r.get("latency_median_ms"),
                "note": METHOD_NOTES.get(name, r.get("note", "")),
            })
    return rows


COLUMNS = [
    ("method", "method", ""),
    ("obj_mean", "obj (mean)", ".6g"),
    ("gap_mean", "gap% mean", ".4g"),
    ("gap_max", "gap% max", ".4g"),
    ("gap_mean_feasible", "gap% mean(feas)", ".4g"),
    ("feas_rate", "feas rate", ".3f"),
    ("domain_rate", "domain valid", ".3f"),
    ("eq_max", "max |h|", ".2e"),
    ("ineq_max", "max viol", ".2e"),
    ("lat_median_ms", "latency ms", ".4g"),
]


def write_report(app: str, tag: str = "") -> str:
    out_dir = os.path.join(DC3_ROOT, "results", app + (f"-{tag}" if tag else ""))
    bench = load_json(os.path.join(out_dir, "benchmark.json"))
    jpath = os.path.join(out_dir, "julia_baselines.json")
    julia = load_json(jpath) if os.path.exists(jpath) else None

    rows = collect_rows(bench, julia)
    table = format_table(rows, COLUMNS)

    corr = bench["dc3"]["correction"]
    lat = bench["dc3"]["latency"]
    env = bench["environment"]
    lines = [
        f"# DC3 benchmark - {app}" + (f" ({tag})" if tag else ""),
        "",
        f"* instances: {bench['dc3']['corrected']['n_instances']} test instances "
        f"(`{os.path.basename(bench['instances_file'])}`)",
        f"* feasibility: exact objective-domain membership and max |h|, max relu(g) <= "
        f"{bench['feas_tol']:g}",
        f"* device `{env['device']}`, dtype `{env['dtype']}`, {env['processor']}, "
        f"torch {env['torch']} ({env['torch_num_threads']} threads)",
        f"* completion: `n_partial` = {bench['partition']['n_partial']}, "
        f"cond(A_D) = {bench['partition']['cond_A_other']:.3e}, "
        f"error amplification ||A_D⁻¹A_P||₂ = {bench['partition'].get('completion_gain', float('nan')):.3e}",
        f"* network parameters: {bench['n_net_params']:,}; training time "
        f"{bench['train_time_s']:.1f} s (excluded from the latency column)",
        "",
        "Quality and headline latency come from the same batch=1 calls on every test instance.",
        "Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.",
        "Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are",
        "undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.",
        "",
        "## Objective, optimality gap and feasibility",
        "",
        "```",
        table,
        "```",
        "",
        "The `CLARABEL(cvxpy)` row is the **reference**: its gap is 0 by definition.  Its",
        "latency includes cvxpy canonicalisation on every call, so it is *not* a fair",
        "solver-speed comparison - use the Julia `Ipopt` rows, which report",
        "`JuMP.solve_time` on a pre-built parametric model, exactly as `problems/` does.",
        "",
        "Gap is `100*|J - J_ref|/|J_ref|` against the reference solver, matching the",
        "convention of `experiments/*/benchmark*.jl`.  `gap% mean(feas)` restricts the",
        "average to instances that pass the feasibility test, so an infeasible point that",
        "undercuts the optimum is not reported as a better solution.",
        "",
        "## Correction",
        "",
        f"* correction steps per timed single-instance solve: mean {corr['steps_mean']:.2f}, "
        f"max {corr['steps_max']} (cap {corr['max_steps_allowed']})",
        f"* instances within `corr_eps` at the end: {100*corr['converged_rate']:.1f}% "
        f"(**{corr['correction_failures']} correction failures**)",
        f"* first step at which an instance became feasible: mean "
        f"{corr['first_feasible_step_mean']:.2f}, max {corr['first_feasible_step_max']:.0f}",
        "",
        "`correction failures` counts instances that did not reach `corr_eps` on DC3's",
        "*internal* criterion (margin-tightened and, for the eco-MPC, row-scaled).",
        "`feas rate` in the table above is measured on the **original** constraints, so",
        "the two numbers can differ. Internal convergence does not certify objective-domain",
        "membership; domain validity is checked separately. A finite step budget never implies",
        "imply feasibility - both numbers are reported.",
        "",
        "## Latency (warm-up + device synchronisation, timing excludes host->device transfer)",
        "",
        "DC3's test-time correction loop is **batch-global**: it keeps stepping until every",
        "instance in the batch is within `corr_eps`.  Large batches therefore pay for their",
        "worst instance, which is why `ms/instance` is *not* monotone in the batch size.",
        "",
    ]
    lat_rows = []
    for k, v in lat.items():
        if not isinstance(v, dict):
            continue
        lat_rows.append({"stage": k, "median_ms": v.get("median_ms"), "mean_ms": v.get("mean_ms"),
                         "p90_ms": v.get("p90_ms"),
                         "per_inst_ms": v.get("per_instance_ms"),
                         "thru": v.get("throughput_inst_per_s")})
    lines += ["```", format_table(lat_rows, [
        ("stage", "stage", ""), ("median_ms", "median ms", ".4g"),
        ("mean_ms", "mean ms", ".4g"), ("p90_ms", "p90 ms", ".4g"),
        ("per_inst_ms", "ms/instance", ".4g"), ("thru", "inst/s", ".5g")]), "```", ""]

    if julia is None:
        lines += [
            "## Julia baselines",
            "",
            "`julia_baselines.json` not present - the LME-ADMM / Ipopt baselines from",
            "`experiments/` were **not executed** for this run.  Produce them with",
            f"`julia --project=. DC3/julia/baselines_{'cone' if 'cone' in app else 'power'}.jl "
            f"DC3/results/{app + (f'-{tag}' if tag else '')}`.",
            "",
        ]
    else:
        lines += ["## Julia baselines", "", julia.get("note", ""), ""]

    md = "\n".join(lines)
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w") as f:
        f.write(md + "\n")
    save_csv(os.path.join(out_dir, "summary.csv"),
             {c[0]: [r.get(c[0]) for r in rows] for c in COLUMNS})
    print(md)
    _plots(app, tag, out_dir, bench, julia)
    return path


def _plots(app, tag, out_dir, bench, julia):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e}; skipping plots)")
        return

    per = os.path.join(out_dir, "per_instance.csv")
    if not os.path.exists(per):
        return
    import csv as _csv

    with open(per) as f:
        rows = list(_csv.DictReader(f))
    gap = np.array([float(r["dc3_gap_pct"]) for r in rows if r.get("dc3_gap_pct")])
    ineq = np.array([float(r["dc3_ineq_max"]) for r in rows if r.get("dc3_ineq_max")])

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].hist(gap, bins=30, color="#4C72B0")
    ax[0].set_xlabel("relative optimality gap (%)")
    ax[0].set_ylabel("instances")
    ax[0].set_title("DC3 optimality gap")
    ax[1].hist(np.log10(np.maximum(ineq, 1e-16)), bins=30, color="#DD8452")
    ax[1].axvline(np.log10(bench["feas_tol"]), color="k", ls="--", label="feas. tol")
    ax[1].set_xlabel("log10 max inequality violation")
    ax[1].set_title("DC3 constraint violation")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dc3_gap_violation.pdf"))
    plt.close(fig)

    lat = bench["dc3"]["latency"]
    labels, med = [], []
    if bench.get("reference"):
        labels.append(bench["reference"]["solver"])
        med.append(bench["reference"]["latency_ms"]["median_ms"])
    labels.append("DC3")
    med.append(lat["single_instance"]["median_ms"])
    if julia:
        for name, r in julia.get("methods", {}).items():
            if r.get("latency_median_ms"):
                labels.append(name)
                med.append(r["latency_median_ms"])
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.bar(range(len(med)), med, color="#55A868")
    ax.set_xticks(range(len(med)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("median single-instance latency (ms)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_comparison.pdf"))
    plt.close(fig)
    print(f"(plots written to {out_dir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=["entr_max", "power_grid"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    write_report(a.app, a.tag)


if __name__ == "__main__":
    main()
