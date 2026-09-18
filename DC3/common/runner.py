"""Application-agnostic training and benchmarking drivers.

Each application supplies an :class:`AppSpec` describing how to build its
problem, its variable partition, its instance splits and its reference solver;
`run_training` and `run_benchmark` then do the same thing for both.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from .completion import LinearCompletion
from .dc3 import DC3Config, DC3Solver, train_dc3
from .io_utils import DC3_ROOT, environment_report, load_json, save_csv, save_json, set_seed
from .metrics import aggregate, format_table, per_instance_metrics
from .problem import ParametricProblem
from .timing import measure, summarize, sync


@dataclass
class AppSpec:
    name: str
    build_problem: Callable[[dict, torch.device, torch.dtype], ParametricProblem]
    partition_other_vars: Callable[[ParametricProblem, dict], Optional[list]]
    build_split: Callable[[dict, str, int, torch.device, torch.dtype], Any]
    index_fn: Callable[[Any, torch.Tensor], Any]
    reference_solve: Callable[[Any, dict], tuple]   # -> (Y, times_s, J, status)
    export_instances: Callable[[Any, str, dict], str]
    default_config: str


# ---------------------------------------------------------------------------
def load_config(spec: AppSpec, path: Optional[str]) -> dict:
    path = path or spec.default_config
    cfg = load_json(path)
    cfg.setdefault("problem", {})
    cfg.setdefault("data", {})
    cfg.setdefault("partition", {"strategy": "explicit"})
    cfg.setdefault("dc3", {})
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def build(spec: AppSpec, cfg: dict):
    dc3cfg = DC3Config.from_dict(cfg["dc3"])
    device = dc3cfg.resolve_device()
    dtype = dc3cfg.torch_dtype()
    set_seed(dc3cfg.seed)
    problem = spec.build_problem(cfg["problem"], device, dtype)
    strategy = cfg["partition"].get("strategy", "explicit")
    other = spec.partition_other_vars(problem, cfg["partition"]) if strategy == "explicit" else None
    comp = LinearCompletion(problem.A_eq, strategy=strategy, other_vars=other,
                            seed=dc3cfg.seed,
                            cond_warn=float(cfg["partition"].get("cond_warn", 1e10)))
    comp.check(strict=bool(cfg["partition"].get("strict", True)))
    comp.to(device, dtype)
    solver = DC3Solver(problem, comp, dc3cfg).to(device=device, dtype=dtype)
    return problem, comp, solver, dc3cfg, device, dtype


@torch.no_grad()
def _fit_input_norm_chunked(solver, problem, spec, params, n, chunk: int = 64) -> None:
    """Streaming mean/std of the network input.

    Materialising the whole feature matrix at once is fine for the eco-MPC
    (193 columns) but not for the cone program at n=1000, m=100, where one
    feature row is 100_100 numbers and the training set is hundreds of MB.
    """
    device = next(solver.parameters()).device
    total = torch.zeros(problem.x_dim, dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    for s0 in range(0, n, chunk):
        idx = torch.arange(s0, min(s0 + chunk, n), device=device)
        X = problem.features(spec.index_fn(params, idx)).double()
        total += X.sum(dim=0)
        total_sq += (X * X).sum(dim=0)
    mean = total / n
    var = (total_sq / n - mean * mean).clamp(min=0)
    std = var.sqrt()
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    solver.net.x_mean.copy_(mean.to(solver.net.x_mean.dtype))
    solver.net.x_std.copy_(std.to(solver.net.x_std.dtype))


def results_dir(spec: AppSpec, cfg: dict, tag: str = "") -> str:
    d = os.path.join(DC3_ROOT, "results", spec.name + (f"-{tag}" if tag else ""))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
def run_training(spec: AppSpec, cfg: dict, tag: str = "", quiet: bool = False) -> dict:
    problem, comp, solver, dc3cfg, device, dtype = build(spec, cfg)
    d = cfg["data"]
    t0 = time.perf_counter()
    train_p = spec.build_split(d, "train", int(d["n_train"]), device, dtype)
    valid_p = spec.build_split(d, "valid", int(d["n_valid"]), device, dtype)
    data_time = time.perf_counter() - t0

    _fit_input_norm_chunked(solver, problem, spec, train_p, len(train_p))
    n_params = solver.net.n_params()
    print(f"[{spec.name}] {comp.info.summary()}")
    print(f"[{spec.name}] x_dim={problem.x_dim}  n_y={problem.n_y}  n_eq={problem.n_eq}  "
          f"n_ineq={problem.n_ineq}  partial={comp.n_partial}  net params={n_params:,}")
    print(f"[{spec.name}] device={device} dtype={dtype} "
          f"train={len(train_p)} valid={len(valid_p)} (data build {data_time:.1f}s)")

    hist = train_dc3(solver, train_p, valid_p, dc3cfg, spec.index_fn, len(train_p),
                     log_fn=(lambda *_: None) if quiet else print)

    out = results_dir(spec, cfg, tag)
    ckpt = os.path.join(out, "checkpoint.pt")
    torch.save({"state_dict": solver.state_dict(), "config": cfg,
                "partition": comp.info.__dict__, "history": hist}, ckpt)
    save_json(os.path.join(out, "train_history.json"), {
        "config": cfg, "history": hist, "n_params": n_params,
        "partition": comp.info.__dict__,
        "data_build_time_s": data_time,
        "environment": environment_report(device, dtype),
    })
    print(f"[{spec.name}] training time {hist['train_time_s']:.1f}s "
          f"(best epoch {hist['best_epoch']}) -> {ckpt}")
    return {"checkpoint": ckpt, "history": hist, "out_dir": out}


# ---------------------------------------------------------------------------
def _reference(spec: AppSpec, cfg: dict, test_p, out_dir: str, force: bool) -> dict:
    path = os.path.join(out_dir, "reference.npz")
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {"Y": z["Y"], "time_s": z["time_s"], "J": z["J"],
                "status": [str(s) for s in z["status"]]}
    print(f"[{spec.name}] solving {len(test_p)} test instances with the reference solver ...")
    t0 = time.perf_counter()
    Y, times, J, status = spec.reference_solve(test_p, cfg)
    print(f"[{spec.name}] reference solve took {time.perf_counter()-t0:.1f}s "
          f"(median {1e3*np.median(times):.2f} ms/instance)")
    np.savez_compressed(path, Y=Y, time_s=times, J=J, status=np.array(status, dtype=object))
    return {"Y": Y, "time_s": times, "J": J, "status": status}


def _dc3_eval(solver: DC3Solver, eval_problem, params, eval_params, J_ref, tol: float) -> dict:
    """Run DC3 in its native precision, then score in float64 on the CPU.

    Scoring in float64 matters when the network runs in float32: the feasibility
    threshold (1e-4) is close enough to float32 round-off on quantities of size
    ~1e2 that residuals must not be measured in the working precision.
    """
    out = solver.solve(params)
    Yc = out["Y"].detach().to(device="cpu", dtype=torch.float64)
    Yr = out["Y_raw"].detach().to(device="cpu", dtype=torch.float64)
    m_corr = per_instance_metrics(eval_problem, eval_params, Yc, J_ref, tol)
    m_raw = per_instance_metrics(eval_problem, eval_params, Yr, J_ref, tol)
    conv = out["converged"].detach().cpu().numpy()
    first = out["first_feasible_step"].detach().cpu().numpy()
    return {
        "corrected": m_corr,
        "raw": m_raw,
        "corr_steps_batch": int(out["steps"]),
        "corr_converged": conv.astype(float),
        "first_feasible_step": first.astype(float),
        "Y": Yc.numpy(),
    }


def _latency(spec: AppSpec, solver: DC3Solver, test_p, device, batch_sizes, n_warmup, n_repeat) -> dict:
    res = {}
    n = len(test_p)
    # single instance (batch = 1): a different instance on every repeat
    one = [spec.index_fn(test_p, torch.tensor([i % n], device=device)) for i in range(min(n, 64))]
    counter = {"i": 0}

    def single():
        p = one[counter["i"] % len(one)]
        counter["i"] += 1
        return solver.solve(p)

    res["single_instance"] = summarize(measure(single, device, n_warmup, n_repeat))
    res["single_instance"]["note"] = "batch=1, one instance per call, cycling through the test set"

    for bs in batch_sizes:
        if bs > n:
            continue
        idx = torch.arange(bs, device=device)
        pb = spec.index_fn(test_p, idx)
        s = summarize(measure(lambda: solver.solve(pb), device, n_warmup, max(10, n_repeat // 4)))
        s["batch_size"] = bs
        s["per_instance_ms"] = s["median_ms"] / bs
        s["throughput_inst_per_s"] = bs / (s["median_ms"] / 1e3)
        res[f"batch_{bs}"] = s

    # stage breakdown at batch = 1
    p1 = one[0]
    with torch.no_grad():
        res["stage_predict_b1"] = summarize(measure(lambda: solver.predict_partial(p1), device, n_warmup, n_repeat))
        Z1 = solver.predict_partial(p1)
        res["stage_complete_b1"] = summarize(measure(lambda: solver.complete(p1, Z1), device, n_warmup, n_repeat))
    res["stage_correct_b1"] = summarize(measure(lambda: solver.correct_test(p1, Z1), device, n_warmup, n_repeat))
    return res


def run_benchmark(spec: AppSpec, cfg: dict, tag: str = "", checkpoint: Optional[str] = None,
                  force_reference: bool = False, n_repeat: int = 50, n_warmup: int = 10,
                  batch_sizes=(1, 8, 32, 100), skip_reference: bool = False) -> dict:
    problem, comp, solver, dc3cfg, device, dtype = build(spec, cfg)
    out_dir = results_dir(spec, cfg, tag)
    ckpt_path = checkpoint or os.path.join(out_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}; run the training entry point first")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    solver.load_state_dict(ck["state_dict"])
    solver.eval()
    train_time = float(ck.get("history", {}).get("train_time_s", float("nan")))

    d = cfg["data"]
    test_p = spec.build_split(d, "test", int(d["n_test"]), device, dtype)
    # float64 / CPU twins used exclusively for scoring (see _dc3_eval)
    cpu64 = torch.device("cpu")
    eval_problem = spec.build_problem(cfg["problem"], cpu64, torch.float64)
    eval_params = spec.build_split(d, "test", int(d["n_test"]), cpu64, torch.float64)
    inst_path = spec.export_instances(eval_params, out_dir, cfg)
    print(f"[{spec.name}] test instances -> {inst_path}")

    tol = dc3cfg.feas_tol
    report: dict[str, Any] = {
        "app": spec.name,
        "config": cfg,
        "partition": comp.info.__dict__,
        "environment": environment_report(device, dtype),
        "feas_tol": tol,
        "train_time_s": train_time,
        "n_net_params": solver.net.n_params(),
        "instances_file": inst_path,
    }

    J_ref = None
    if not skip_reference:
        ref = _reference(spec, cfg, eval_params, out_dir, force_reference)
        J_ref = np.asarray(ref["J"], dtype=float)
        report["reference"] = {
            "solver": cfg.get("reference", {}).get("solver", "CLARABEL"),
            "status_counts": {s: int(sum(1 for x in ref["status"] if x == s)) for s in set(ref["status"])},
            "latency_ms": summarize(np.asarray(ref["time_s"])),
            "obj_mean": float(np.nanmean(J_ref)),
        }
        Yref = torch.as_tensor(ref["Y"], dtype=torch.float64, device=cpu64)
        report["reference"]["feasibility"] = aggregate(
            per_instance_metrics(eval_problem, eval_params, Yref, J_ref, tol), tol)

    ev = _dc3_eval(solver, eval_problem, test_p, eval_params, J_ref, tol)
    report["dc3"] = {
        "corrected": aggregate(ev["corrected"], tol),
        "raw_no_correction": aggregate(ev["raw"], tol),
        "correction": {
            "batch_steps_used": ev["corr_steps_batch"],
            "max_steps_allowed": dc3cfg.corr_test_max_steps,
            "converged_rate": float(np.mean(ev["corr_converged"])),
            "correction_failures": int(np.sum(ev["corr_converged"] < 0.5)),
            "first_feasible_step_mean": float(np.mean(ev["first_feasible_step"][ev["first_feasible_step"] >= 0]))
            if np.any(ev["first_feasible_step"] >= 0) else float("nan"),
            "first_feasible_step_max": float(np.max(ev["first_feasible_step"])),
            "note": "DC3's test-time loop is batch-global: it stops when every instance "
                    "in the batch is within corr_eps, so batch_steps_used is a batch quantity.",
        },
    }

    print(f"[{spec.name}] measuring inference latency ...")
    report["dc3"]["latency"] = _latency(spec, solver, test_p, device, batch_sizes, n_warmup, n_repeat)

    # per-instance dump for downstream plots / tables
    cols = {f"dc3_{k}": v for k, v in ev["corrected"].items() if isinstance(v, np.ndarray)}
    cols["dc3_corr_converged"] = ev["corr_converged"]
    if J_ref is not None:
        cols["J_ref"] = J_ref
    save_csv(os.path.join(out_dir, "per_instance.csv"), cols)
    np.savez_compressed(os.path.join(out_dir, "dc3_solutions.npz"), Y=ev["Y"])

    save_json(os.path.join(out_dir, "benchmark.json"), report)
    print(f"[{spec.name}] wrote {os.path.join(out_dir, 'benchmark.json')}")
    return report


# ---------------------------------------------------------------------------
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--set", type=str, nargs="*", default=[],
                    help="config overrides, e.g. --set dc3.epochs=5 problem.n=100")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    return cfg
