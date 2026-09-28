"""Solving-time / optimality-gap table for the maximum-entropy cone program:
the DC3 + correction column.  The IPOPT and sLME-ADMM columns are filled by
table.jl, which also renders the table.

Run from the repository root with the DC3 environment (see DC3/README.md)::

    python experiments/entr_max/table.py            # use stored data, run what is missing
    python experiments/entr_max/table.py --force    # re-evaluate DC3 on every size
    python experiments/entr_max/table.py --retrain  # also retrain the DC3 networks

DC3 is evaluated on the same 1000 instances per (n, m) as IPOPT and sLME-ADMM
(results/entr_max/table/instances/, written by table.jl, which is called here if
they do not exist yet).  The gap is measured against the Ipopt (tol 1e-8)
ground truth stored in results/entr_max/table/ground_truth-n=..-m=...npz.  DC3 saves only
its solutions `W`; they are scored by evaluate.jl with src/metrics.jl, the
same functions that score IPOPT and sLME-ADMM.
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

from DC3.common.runner import apply_overrides, build, load_config, run_training  # noqa: E402
from DC3.common.timing import sync  # noqa: E402
from DC3.entr_max import data as D  # noqa: E402
from DC3.entr_max.experiment import CONFIG_DIR, SPEC  # noqa: E402

OUT = os.path.join(REPO, "results", "entr_max", "table")
INST_DIR = os.path.join(OUT, "instances")
SIZES = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
N_WARMUP = 10
DC3_VERSION = 4                # bump to invalidate stored DC3 solutions
METRICS_VERSION = 4            # must equal METRICS_VERSION in src/metrics.jl

# DC3 networks that already exist in DC3/results (see DC3/entr_max/README.md);
# other sizes are trained with the settings of the config for the same n.
EXISTING_TAGS = {(100, 10): "small", (1000, 100): "default"}
BASE_CONFIG = {100: "small.json", 1000: "default.json"}


def inst_file(n, m):
    return os.path.join(INST_DIR, f"instances-n={n}-m={m}.npz")


def gt_file(n, m):
    return os.path.join(OUT, f"ground_truth-n={n}-m={m}.npz")


def dc3_file(n, m):
    return os.path.join(OUT, f"DC3-n={n}-m={m}.npz")


# ---------------------------------------------------------------------------
def julia(script: str, *args: str) -> None:
    subprocess.run(["julia", f"--project={REPO}", "--threads=auto",
                    os.path.join(os.path.dirname(__file__), script), *args], check=True, cwd=REPO)


def ensure_instances(n: int, m: int) -> None:
    if os.path.exists(inst_file(n, m)) and os.path.exists(gt_file(n, m)):
        return
    julia("table.jl", "worker", str(n), str(m), "--instances-only")


def checkpoint(n: int, m: int, retrain: bool) -> dict:
    tag = EXISTING_TAGS.get((n, m), f"table-n{n}-m{m}")
    path = os.path.join(REPO, "DC3", "results", f"entr_max-{tag}", "checkpoint.pt")
    if os.path.exists(path) and not retrain:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = ck["config"]["problem"]
        if (int(p["n"]), int(p["m"])) == (n, m):
            return ck
    cfg = load_config(SPEC, os.path.join(CONFIG_DIR, BASE_CONFIG[n]))
    cfg = apply_overrides(cfg, [f"problem.n={n}", f"problem.m={m}", f"data.n={n}", f"data.m={m}"])
    print(f"[entr_max_table] training DC3 for n={n}, m={m} (tag {tag}) ...")
    run_training(SPEC, cfg, tag=tag, quiet=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(n: int, m: int, ck: dict) -> None:
    cfg = ck["config"]
    _, _, solver, _, device, dtype = build(SPEC, cfg)
    solver.load_state_dict(ck["state_dict"])
    solver.eval()

    inst = np.load(inst_file(n, m))
    A, b = inst["A"], inst["b"]
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

    W = torch.cat(Y).numpy()
    np.savez(dc3_file(n, m), W=W, time_ms=t_ms, corr_steps=steps, dc3_version=DC3_VERSION)
    print(f"[entr_max_table] n={n} m={m} DC3 + correction: time {t_ms.mean():.3f} ({t_ms.max():.3f}) ms, "
          f"{steps.mean():.1f} correction steps")
    julia("evaluate.jl", str(n), str(m))


def is_current(n: int, m: int) -> bool:
    if not os.path.exists(dc3_file(n, m)):
        return False
    with np.load(dc3_file(n, m)) as d:
        return (d.get("dc3_version", 0) == DC3_VERSION and "gap_pct" in d
                and d.get("metrics_version", 0) == METRICS_VERSION)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate DC3 even if its data exists")
    ap.add_argument("--retrain", action="store_true", help="retrain the DC3 networks (implies --force)")
    a = ap.parse_args()
    for n, m in SIZES:
        if is_current(n, m) and not (a.force or a.retrain):
            continue
        ensure_instances(n, m)
        evaluate(n, m, checkpoint(n, m, a.retrain))
    julia("table.jl")                   # runs what IPOPT / sLME-ADMM data is missing, renders the table


if __name__ == "__main__":
    main()
