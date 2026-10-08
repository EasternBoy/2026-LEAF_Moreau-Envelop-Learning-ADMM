"""DC3 + correction column for the QP + L1 table, following experiments/qp/table.py.

    DC3/.venv/bin/python experiments/qp_l1/table.py
    DC3/.venv/bin/python experiments/qp_l1/table.py --force     # re-evaluate DC3
    DC3/.venv/bin/python experiments/qp_l1/table.py --retrain   # retrain DC3 (implies --force)

Trains DC3 (tag default) with the tuned configuration DC3/results/qp_l1-tune/best.json when
present (python -m DC3.tune --app qp_l1 ...), otherwise DC3/qp_l1/configs/default.json.
Evaluates it at batch size 1 on the saved Julia inputs and writes
results/qp_l1/table/DC3.npz; experiments/qp_l1/table.jl --render scores and renders all methods.
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
from DC3.qp_l1 import data as D  # noqa: E402
from DC3.qp_l1.experiment import SPEC  # noqa: E402

OUT = os.path.join(REPO, "results", "qp_l1", "table")
INSTANCES = os.path.join(OUT, "instances.npz")
DC3_FILE = os.path.join(OUT, "DC3.npz")
TUNED = os.path.join(REPO, "DC3", "results", "qp_l1-tune", "best.json")
N_WARMUP = 10


def julia(*args):
    subprocess.run(["julia", f"--project={REPO}", os.path.join(os.path.dirname(__file__), "table.jl"), *args],
                   check=True, cwd=REPO)


def checkpoint(retrain):
    path = os.path.join(REPO, "DC3", "results", "qp_l1-default", "checkpoint.pt")
    if not os.path.exists(path) or retrain:
        config = TUNED if os.path.exists(TUNED) else None
        print(f"[qp_l1_table] training DC3 (tag default, config {config or 'default'}) ...")
        run_training(SPEC, load_config(SPEC, config), tag="default", quiet=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(ck):
    _, _, solver, _, device, dtype = build(SPEC, ck["config"])
    solver.load_state_dict(ck["state_dict"])
    solver.eval()
    with np.load(INSTANCES) as saved:
        X = saved["X"]
        for key, expected in D.fixed_data().items():
            if not np.allclose(saved[key], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Julia QP + L1 matrix {key} differs from DC3")
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
    dc = ck["config"]["dc3"]
    np.savez(DC3_FILE, W=torch.cat(Y).numpy(), time_ms=times, corr_steps=steps,
             epochs=dc["epochs"], corr_lr=dc["corr_lr"], corr_test_max_steps=dc["corr_test_max_steps"])
    print(f"[qp_l1_table] DC3 + correction: {times.mean():.3f} ({times.max():.3f}) ms, "
          f"{steps.mean():.1f} correction steps; device={device}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate DC3 even if saved results exist")
    ap.add_argument("--retrain", action="store_true", help="retrain DC3 (implies --force)")
    args = ap.parse_args()
    if not os.path.exists(INSTANCES):
        julia()
    if not os.path.exists(DC3_FILE) or args.force or args.retrain:
        evaluate(checkpoint(args.retrain))
    julia("--render")


if __name__ == "__main__":
    main()
