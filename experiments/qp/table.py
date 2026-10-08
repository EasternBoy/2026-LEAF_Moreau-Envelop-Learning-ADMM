"""DC3 + correction column for the QP table, following experiments/entr_max/table.py.

    python3 experiments/qp/table.py
    python3 experiments/qp/table.py --force
    python3 experiments/qp/table.py --retrain

Evaluate DC3 at batch size 1 on the saved Julia inputs. Missing Julia results and
DC3 checkpoints are generated as in the entropy script. evaluate.jl scores all
solutions with table.jl's score function and renders the three-method table.
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

from DC3.common.runner import build, load_config, run_training  # noqa: E402
from DC3.common.timing import sync  # noqa: E402
from DC3.qp import data as D  # noqa: E402
from DC3.qp.experiment import SPEC  # noqa: E402

OUT = os.path.join(REPO, "results", "qp", "table")
SUFFIX = "n=100-neq=50-m=50-samples=1000"
OSQP_FILE = f"OSQP-{SUFFIX}-tol=1.0e-8.npz"
SLME_FILE = f"sLME-ADMM-{SUFFIX}-tol=0.001-gopt=1.0.npz"
DC3_FILE = os.path.join(OUT, f"DC3-{SUFFIX}.npz")
N_WARMUP = 10
DC3_VERSION = 1


def julia(script):
    subprocess.run(["julia", f"--project={REPO}",
                    os.path.join(os.path.dirname(__file__), script)], check=True, cwd=REPO)


def ensure_instances():
    paths = [os.path.join(OUT, "instances", f"instances-{SUFFIX}.npz"),
             os.path.join(OUT, f"ground_truth-{SUFFIX}.npz"),
             os.path.join(OUT, OSQP_FILE), os.path.join(OUT, SLME_FILE)]
    if all(os.path.exists(path) for path in paths):
        # Replace earlier parallel-workload timings with the current sequential run.
        for path in paths[2:]:
            with np.load(path) as saved:
                if "total_time_s" in saved:
                    break
        else:
            return
    julia("table.jl")


def checkpoint(retrain):
    path = os.path.join(REPO, "DC3", "results", "qp-default", "checkpoint.pt")
    if not os.path.exists(path) or retrain:
        print("[qp_table] training DC3 (tag default) ...")
        run_training(SPEC, load_config(SPEC, None), tag="default", quiet=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(ck):
    _, _, solver, _, device, dtype = build(SPEC, ck["config"])
    solver.load_state_dict(ck["state_dict"])
    solver.eval()
    with np.load(os.path.join(OUT, "instances", f"instances-{SUFFIX}.npz")) as saved:
        X = saved["X"]
        for key, expected in D.fixed_data().items():
            if not np.allclose(saved[key], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Julia QP matrix {key} differs from DC3")
    params = D.to_params(X, device, dtype)
    one = [params.index(torch.tensor([i], device=device)) for i in range(len(X))]
    for _ in range(N_WARMUP):
        solver.solve(one[0])
    times = np.empty(len(X))
    steps = np.empty(len(X), dtype=int)
    Y = []
    for i, p in enumerate(one):
        sync(device)
        start = time.perf_counter()
        result = solver.solve(p)
        sync(device)
        times[i] = 1e3 * (time.perf_counter() - start)
        steps[i] = result["steps"]
        Y.append(result["Y"].detach().to("cpu", torch.float64))
    np.savez(DC3_FILE, W=torch.cat(Y).numpy(), time_ms=times, corr_steps=steps,
             dc3_version=DC3_VERSION)
    print(f"[qp_table] DC3 + correction: {times.mean():.3f} ({times.max():.3f}) ms, "
          f"{steps.mean():.1f} correction steps; device={device}")


def is_current():
    if not os.path.exists(DC3_FILE):
        return False
    with np.load(DC3_FILE) as saved:
        return saved.get("dc3_version", 0) == DC3_VERSION and "gap_pct" in saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate DC3 even if saved results exist")
    ap.add_argument("--retrain", action="store_true", help="retrain DC3 (implies --force)")
    args = ap.parse_args()
    ensure_instances()
    if not is_current() or args.force or args.retrain:
        evaluate(checkpoint(args.retrain))
    julia("evaluate.jl")


if __name__ == "__main__":
    main()
