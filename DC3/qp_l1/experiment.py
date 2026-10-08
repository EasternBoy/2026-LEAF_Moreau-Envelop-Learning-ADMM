"""AppSpec wiring, following DC3/entr_max/experiment.py."""

from __future__ import annotations

import os

import numpy as np

from ..common.runner import AppSpec
from . import data as D
from . import reference as R
from .problem import QPProblem, index_qp

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def build_problem(pc, device, dtype):
    return QPProblem(**D.fixed_data(), lam=D.LAMBDA, device=device, dtype=dtype)


def partition_other_vars(prob, pcfg):
    return list(pcfg.get("other_vars", range(prob.n_eq)))


def build_split(dc, split, count, device, dtype):
    return D.make_split(count, int(dc["seed"]), split, dc["test_instances"], device, dtype)


def reference_solve(params, cfg):
    rc = cfg.get("reference", {})
    return R.solve_batch(params.x.detach().cpu().double().numpy(),
                         solver=rc.get("solver", "CLARABEL"), tol=float(rc.get("tol", 1e-9)))


def export_instances(params, out_dir, cfg):
    path = os.path.join(out_dir, "test_instances.npz")
    np.savez_compressed(path, X=params.x.detach().cpu().double().numpy(), lam=D.LAMBDA, **D.fixed_data())
    return path


SPEC = AppSpec(
    name="qp_l1",
    build_problem=build_problem,
    partition_other_vars=partition_other_vars,
    build_split=build_split,
    index_fn=index_qp,
    reference_solve=reference_solve,
    export_instances=export_instances,
    default_config=os.path.join(CONFIG_DIR, "default.json"),
)
