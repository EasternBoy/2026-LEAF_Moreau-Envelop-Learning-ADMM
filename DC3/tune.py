"""Small grid search over DC3 hyper-parameters, scored on the **validation** split.

    python -m DC3.tune --app power_grid --grid '{"dc3.corr_lr":[1e-3,1e-2],"dc3.lr":[1e-3]}'

Test instances are never touched.  Selection is lexicographic: highest
validation feasible rate first (ties broken by 1e-3 relative), then lowest mean
validation objective; the winner is written to ``results/<app>-tune/best.json``
so it can be fed back with ``--config``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time

import numpy as np
import torch

from .common.dc3 import DC3Config, train_dc3
from .common.io_utils import DC3_ROOT, save_json
from .common.metrics import aggregate, per_instance_metrics
from .common.runner import _fit_input_norm_chunked, apply_overrides, build, load_config

APPS = {"entr_max": "DC3.entr_max.experiment", "qp_l1": "DC3.qp_l1.experiment",
        "power_grid": "DC3.power_grid.experiment"}


def evaluate_valid(spec, cfg) -> dict:
    problem, comp, solver, dc3cfg, device, dtype = build(spec, cfg)
    d = cfg["data"]
    train_p = spec.build_split(d, "train", int(d["n_train"]), device, dtype)
    valid_p = spec.build_split(d, "valid", int(d["n_valid"]), device, dtype)
    _fit_input_norm_chunked(solver, problem, spec, train_p, len(train_p))
    t0 = time.perf_counter()
    hist = train_dc3(solver, train_p, valid_p, dc3cfg, spec.index_fn, len(train_p),
                     log_fn=lambda *_: None)
    train_time = time.perf_counter() - t0

    out = solver.solve(valid_p)
    cpu64 = torch.device("cpu")
    ep = spec.build_problem(cfg["problem"], cpu64, torch.float64)
    vp = spec.build_split(d, "valid", int(d["n_valid"]), cpu64, torch.float64)
    m = per_instance_metrics(ep, vp, out["Y"].detach().to(cpu64, torch.float64),
                             None, dc3cfg.feas_tol)
    a = aggregate(m, dc3cfg.feas_tol)
    return {
        "valid_feasible_rate": a["feasible_rate"],
        "valid_obj_mean": a["obj_mean"],
        "valid_ineq_max": a["ineq_max_max"],
        "valid_corr_steps": int(out["steps"]),
        "valid_corr_converged": float(out["converged"].double().mean().item()),
        "train_time_s": train_time,
        "best_epoch": hist["best_epoch"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=list(APPS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", required=True, help="JSON dict of dotted-key -> list of values")
    ap.add_argument("--set", nargs="*", default=[], help="fixed overrides applied to every point")
    ap.add_argument("--tag", default="tune")
    a = ap.parse_args()

    import importlib
    spec = importlib.import_module(APPS[a.app]).SPEC
    base = apply_overrides(load_config(spec, a.config), a.set)
    grid = json.loads(a.grid)
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = apply_overrides(copy.deepcopy(base), [f"{k}={json.dumps(v)}" for k, v in zip(keys, combo)])
        point = dict(zip(keys, combo))
        print(f"--- {point}")
        try:
            r = evaluate_valid(spec, cfg)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            r = {"error": f"{type(e).__name__}: {e}"}
        r["point"] = point
        print("   ", {k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in r.items() if k != "point"})
        results.append(r)

    ok = [r for r in results if "error" not in r]
    if not ok:
        raise SystemExit("every configuration failed")
    best_feas = max(r["valid_feasible_rate"] for r in ok)
    cand = [r for r in ok if r["valid_feasible_rate"] >= best_feas - 1e-3]
    best = min(cand, key=lambda r: r["valid_obj_mean"])
    out_dir = os.path.join(DC3_ROOT, "results", f"{a.app}-{a.tag}")
    best_cfg = apply_overrides(copy.deepcopy(base),
                               [f"{k}={json.dumps(v)}" for k, v in best["point"].items()])
    best_cfg.pop("_config_path", None)
    save_json(os.path.join(out_dir, "tune_results.json"),
              {"grid": grid, "fixed": a.set, "results": results, "best": best})
    save_json(os.path.join(out_dir, "best.json"), best_cfg)
    print(f"\nbest point: {best['point']}  "
          f"(valid feas {best['valid_feasible_rate']:.3f}, obj {best['valid_obj_mean']:.6g})")
    print(f"written to {out_dir}/best.json")


if __name__ == "__main__":
    main()
