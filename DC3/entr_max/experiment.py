"""AppSpec wiring for the maximum-entropy cone program."""

from __future__ import annotations

import os

import numpy as np
import torch

from ..common.runner import AppSpec
from . import data as D
from . import reference as R
from .problem import MaxEntropyProblem, index_cone

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def build_problem(pc: dict, device, dtype) -> MaxEntropyProblem:
    return MaxEntropyProblem(
        n=int(pc["n"]), m=int(pc["m"]),
        feature_mode=pc.get("feature_mode", "A_flat_b"),
        w_floor=pc.get("w_floor", None),
        w_margin=float(pc.get("w_margin", 0.0)),
        ineq_margin=float(pc.get("ineq_margin", 0.0)),
        dtype=dtype, device=device,
    )


def partition_other_vars(prob: MaxEntropyProblem, pcfg: dict):
    """The single dependent variable.  Default: the last coordinate."""
    return list(pcfg.get("other_vars", [prob.n - 1]))


def build_split(dc: dict, split: str, count: int, device, dtype):
    if split == "train" and dc.get("stream_train", False):
        return D.StreamingConeParams(int(dc["n"]), int(dc["m"]), count, int(dc["seed"]), split, device, dtype)
    return D.make_split(int(dc["n"]), int(dc["m"]), count, int(dc["seed"]), split, device, dtype)


def reference_solve(params, cfg: dict):
    rc = cfg.get("reference", {})
    A = params.A.detach().cpu().double().numpy()
    b = params.b.detach().cpu().double().numpy()
    w_floor = 0.0 if rc.get("ignore_w_floor", False) else build_problem(cfg["problem"], torch.device("cpu"), torch.float64).w_floor
    return R.solve_batch(A, b, w_floor=w_floor,
                         solver=rc.get("solver", "CLARABEL"),
                         tol=float(rc.get("tol", 1e-9)))


def export_instances(params, out_dir: str, cfg: dict) -> str:
    A = params.A.detach().cpu().double().numpy()
    b = params.b.detach().cpu().double().numpy()
    path = os.path.join(out_dir, "test_instances.npz")
    D.save_test_set(path, A, b, {"n": A.shape[2], "m": A.shape[1], "count": A.shape[0],
                                 "seed": int(cfg["data"]["seed"]), "b_divisor": D.B_DIVISOR})
    return path


SPEC = AppSpec(
    name="entr_max",
    build_problem=build_problem,
    partition_other_vars=partition_other_vars,
    build_split=build_split,
    index_fn=index_cone,
    reference_solve=reference_solve,
    export_instances=export_instances,
    default_config=os.path.join(CONFIG_DIR, "default.json"),
)
