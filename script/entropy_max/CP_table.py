"""Solving-time / optimality-gap table for the maximum-entropy cone program:
the DC3 column.  The IPOPT and sLME-ADMM columns are filled by CP_table.jl.

Run from the repository root with the DC3 environment (see DC3/README.md)::

    python script/entropy_max/CP_table.py            # use stored data, run what is missing
    python script/entropy_max/CP_table.py --force    # re-evaluate DC3 on every size
    python script/entropy_max/CP_table.py --retrain  # also retrain the DC3 networks

DC3 is evaluated on the same 1000 instances per (n, m) as IPOPT and sLME-ADMM
(data/cone_result/instances/, written by CP_table.jl, which is called here if
they do not exist yet).  The gap is measured against the Ipopt (tol 1e-8)
ground truth stored in data/cone_result/ground_truth-n=..-m=...npz.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from DC3.common.metrics import per_instance_metrics  # noqa: E402
from DC3.common.runner import apply_overrides, build, load_config, run_training  # noqa: E402
from DC3.common.timing import sync  # noqa: E402
from DC3.cone_programming import data as D  # noqa: E402
from DC3.cone_programming.experiment import CONFIG_DIR, SPEC  # noqa: E402

OUT = os.path.join(REPO, "data", "cone_result")
INST_DIR = os.path.join(OUT, "instances")
SIZES = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
GAP_ROWS = [(100, 10), (1000, 100)]
GOPTS = [(1.0, "1"), (0.1, "0.1")]            # g_opt in %, and its spelling in file names
N_SAMPLES = 1000
FEAS_TOL = 1e-4
N_WARMUP = 10

# DC3 networks that already exist in DC3/results (see DC3/cone_programming/README.md);
# other sizes are trained with the settings of the config for the same n.
EXISTING_TAGS = {(100, 10): "small", (1000, 100): "default"}
BASE_CONFIG = {100: "small.json", 1000: "default.json"}


def inst_file(n, m):
    return os.path.join(INST_DIR, f"instances-n={n}-m={m}.npz")


def gt_file(n, m):
    return os.path.join(OUT, f"ground_truth-n={n}-m={m}.npz")


def res_file(meth, g, n, m):
    return os.path.join(OUT, f"{meth}-gopt={g}-n={n}-m={m}.npz")


def dc3_file(n, m):
    return os.path.join(OUT, f"DC3-n={n}-m={m}.npz")


# ---------------------------------------------------------------------------
def ensure_instances(n: int, m: int) -> None:
    if os.path.exists(inst_file(n, m)) and os.path.exists(gt_file(n, m)):
        return
    subprocess.run(["julia", f"--project={REPO}", "--threads=auto",
                    os.path.join(os.path.dirname(__file__), "CP_table.jl"),
                    "worker", str(n), str(m), "--instances-only"], check=True, cwd=REPO)


def checkpoint(n: int, m: int, retrain: bool) -> dict:
    tag = EXISTING_TAGS.get((n, m), f"table-n{n}-m{m}")
    path = os.path.join(REPO, "DC3", "results", f"cone_programming-{tag}", "checkpoint.pt")
    if os.path.exists(path) and not retrain:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = ck["config"]["problem"]
        if (int(p["n"]), int(p["m"])) == (n, m):
            return ck
    cfg = load_config(SPEC, os.path.join(CONFIG_DIR, BASE_CONFIG[n]))
    cfg = apply_overrides(cfg, [f"problem.n={n}", f"problem.m={m}", f"data.n={n}", f"data.m={m}"])
    print(f"[CP_table] training DC3 for n={n}, m={m} (tag {tag}) ...")
    run_training(SPEC, cfg, tag=tag, quiet=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(n: int, m: int, ck: dict) -> None:
    cfg = ck["config"]
    _, _, solver, _, device, dtype = build(SPEC, cfg)
    solver.load_state_dict(ck["state_dict"])
    solver.eval()

    inst = np.load(inst_file(n, m))
    A, b = inst["A"], inst["b"]
    J_opt = np.load(gt_file(n, m))["J_opt"]
    params = D.to_params(A, b, device, dtype)
    one = [params.index(torch.tensor([i], device=device)) for i in range(len(A))]

    for _ in range(N_WARMUP):
        solver.solve(one[0])
    t_ms = np.empty(len(A))
    steps = np.empty(len(A))
    Y = []
    for i, p in enumerate(one):                  # batch = 1: one instance per call
        sync(device)
        t0 = time.perf_counter()
        out = solver.solve(p)
        sync(device)
        t_ms[i] = 1e3 * (time.perf_counter() - t0)
        steps[i] = out["steps"]
        Y.append(out["Y"].detach().to("cpu", torch.float64))

    cpu64 = torch.device("cpu")
    eval_problem = SPEC.build_problem(cfg["problem"], cpu64, torch.float64)
    met = per_instance_metrics(eval_problem, D.to_params(A, b, cpu64, torch.float64),
                               torch.cat(Y), J_opt, FEAS_TOL)
    viol = np.maximum.reduce([met["eq_max"], met["ineq_max"], met["domain_max"], np.zeros(len(A))])
    np.savez(dc3_file(n, m), time_ms=t_ms, gap_pct=met["gap_pct"], max_viol=viol,
             feasible=met["feasible"], domain_valid=met["domain_valid"],
             metrics_version=2, corr_steps=steps)
    print(f"[CP_table] n={n} m={m} DC3: time {t_ms.mean():.3f} ({t_ms.max():.3f}) ms, "
          f"gap {met['gap_pct'].mean():.3g} ({met['gap_pct'].max():.3g}) %, "
          f"feasible {met['feasible'].mean():.3f}")


# ---------------------------------------------------------------------------
def fmt_time(x):
    return "%.2f (%.2f)" % (np.mean(x), np.max(x))


def fmt_gap(x):
    if not np.isfinite(x).all():
        return "undefined (outside objective domain)"
    return "%.3g (%.3g)" % (np.mean(x), np.max(x))


def fmt_viol(x):
    return "%.1e (%.1e)" % (np.mean(x), np.max(x))


def time_cell(meth, gname, n, m):
    if not os.path.exists(res_file(meth, gname, n, m)):
        return "—", np.inf
    d = np.load(res_file(meth, gname, n, m))
    if d.get("metrics_version", 0) != 2:
        return "rerun required", np.inf
    if not (np.all(d["feasible"]) and np.all(d["gap_pct"] <= float(gname))):
        return "unable to achieve", np.inf
    t = d["time_ms"]
    return fmt_time(t), float(np.mean(t))


def dc3_cell(n, m, gopt):
    if not os.path.exists(dc3_file(n, m)):
        return "—", np.inf
    d = np.load(dc3_file(n, m))
    if d.get("metrics_version", 0) != 2:
        return "rerun required", np.inf
    ok = bool(np.all(d["feasible"] > 0.5)) and float(np.max(d["gap_pct"])) <= gopt
    return (fmt_time(d["time_ms"]), float(np.mean(d["time_ms"]))) if ok else ("unable to achieve", np.inf)


def iter_suffix(gname, n, m):
    f = res_file("sLME-ADMM", gname, n, m)
    return " [%.1f it.]" % np.mean(np.load(f)["iterations"]) if os.path.exists(f) else ""


def render() -> None:
    rows = []
    for gopt, gname in GOPTS:
        for n, m in SIZES:
            cells = [time_cell("IPOPT", gname, n, m), time_cell("sLME-ADMM", gname, n, m), dc3_cell(n, m, gopt)]
            best = int(np.argmin([c[1] for c in cells]))
            txt = [f"**{c[0]}**" if np.isfinite(c[1]) and i == best else c[0] for i, c in enumerate(cells)]
            txt[1] += iter_suffix(gname, n, m)
            rows.append(f"| solving time (g_opt ≤ {gname}%) | {n} | {m} | {' | '.join(txt)} |")
    for n, m in GAP_ROWS:
        ip = "0" if os.path.exists(gt_file(n, m)) else "—"
        f = res_file("sLME-ADMM", "0.1", n, m)
        sl = "—"
        if os.path.exists(f):
            saved = np.load(f)
            sl = fmt_gap(saved["gap_pct"]) if saved.get("metrics_version", 0) == 2 else "rerun required"
        dc = "—"
        if os.path.exists(dc3_file(n, m)):
            d = np.load(dc3_file(n, m))
            dc = fmt_gap(d["gap_pct"]) if d.get("metrics_version", 0) == 2 else "rerun required"
            feas = float(np.mean(d["feasible"] > 0.5))
            if feas < 1:
                dc += ", feasible %.0f%%" % (100 * feas)
        rows.append(f"| Opt. gap (%) | {n} | {m} | {ip} | {sl} | {dc} |")
    for _, gname in GOPTS:
        for n, m in SIZES:
            cells = [(fmt_viol(np.load(f)["max_viol"]) if np.load(f).get("metrics_version", 0) == 2
                      else "rerun required") if os.path.exists(f) else "—"
                     for f in (res_file("IPOPT", gname, n, m), res_file("sLME-ADMM", gname, n, m), dc3_file(n, m))]
            rows.append(f"| Constr. viol. (g_opt ≤ {gname}%) | {n} | {m} | {' | '.join(cells)} |")
    body = "\n".join(rows)
    md = f"""# Maximum-entropy cone program: solving time and optimality gap

Generated by `script/entropy_max/CP_table.jl` (IPOPT, sLME-ADMM) and
`script/entropy_max/CP_table.py` (DC3) from the data in this folder — see
[README.md](README.md) for what each number means.  IPOPT and sLME-ADMM use oracle-assisted stopping against the known optimum;
reference-solve cost is excluded. DC3 does not use an optimum oracle.
Feasibility additionally requires exact entropy-domain membership (w >= 0).
Undefined entropy gaps are not replaced by a clamped objective.
Solving time in ms and
optimality gap in %, as mean (max) over {N_SAMPLES} instances per row; **bold** is
the lowest mean time in the row; `[k it.]` is sLME-ADMM's mean number of
iterations; — means the data has not been produced yet.  Constr. viol. is the
largest violation `max(max(A w − b), max(−w), |1ᵀw − 1|)` of each returned point
(IPOPT and sLME-ADMM from the run with that g_opt; DC3 has a single run).

|  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 mean (max) |
|---|---|---|---|---|---|
{body}
"""
    with open(os.path.join(OUT, "CP_table.md"), "w") as fh:
        fh.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate DC3 even if its data exists")
    ap.add_argument("--retrain", action="store_true", help="retrain the DC3 networks (implies --force)")
    a = ap.parse_args()
    for n, m in SIZES:
        if os.path.exists(dc3_file(n, m)) and not (a.force or a.retrain):
            with np.load(dc3_file(n, m)) as saved:
                if saved.get("metrics_version", 0) == 2:
                    continue
        ensure_instances(n, m)
        evaluate(n, m, checkpoint(n, m, a.retrain))
    render()


if __name__ == "__main__":
    main()
