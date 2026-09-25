"""Benchmark DC3 against CLARABEL on saved power-grid table inputs.

Set ``N`` below, train a checkpoint for that horizon, and run from the
repository root::

    python3 -m DC3.power_grid.benchmark.generate_table_instances --N 96
    python3 -m DC3.power_grid.benchmark.train_table --tag table-N96-v2
    python3 -m DC3.power_grid.benchmark.table_benchmark --checkpoint DC3/results/power_grid-table-N96-v2/checkpoint.pt

The default input is the single test_instances_*.npz file in
data/solving_data/power_table_inputs/. Inference runs on CPU, one instance at a
time, with at most ``MAX_ITER`` DC3 correction iterations. The first call is a discarded
warm-up. The timed interval is solver.solve(params), with params already on CPU.
CLARABEL references are solved from the same saved inputs, outside DC3 timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...common.metrics import per_instance_metrics
from ...common.runner import build
from ..data import to_params
from ..experiment import SPEC, build_problem
from ..reference import solve_instance


REPO_ROOT = Path(__file__).resolve().parents[3]
N = 96
INSTANCE_DIR = REPO_ROOT / "data" / "solving_data" / "power_table_inputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MAX_ITER = 500


def find_instances(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()
    path = INSTANCE_DIR / f"test_instances_N={N}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run python3 -m DC3.power_grid.benchmark.generate_table_instances --N {N}"
        )
    return path


def load_instances(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(path) as data:
        required = {"N", "x0", "load", "gen", "case_tag_utf8"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        n = int(np.asarray(data["N"]).item())
        x0 = np.asarray(data["x0"], dtype=np.float64)
        load = np.asarray(data["load"], dtype=np.float64)
        gen = np.asarray(data["gen"], dtype=np.float64)
        case_tag = np.asarray(data["case_tag_utf8"], dtype=np.uint8).tobytes().decode("utf-8")
    if n != N or x0.ndim != 1 or x0.size == 0:
        raise ValueError(f"Expected a nonempty N={N} test set in {path}")
    if load.shape != (x0.size, n) or gen.shape != load.shape:
        raise ValueError("x0, load, and gen have inconsistent shapes")
    if not all(np.isfinite(a).all() for a in (x0, load, gen)):
        raise ValueError("Instances must be finite")
    if (case_tag != f"N={N}" and not case_tag.startswith(f"N={N}_")) or path.name != f"test_instances_{case_tag}.npz":
        raise ValueError(f"Case tag does not match the instance filename: {path.name}")
    return x0, load, gen, case_tag


def load_solver(checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Run `python3 -m DC3.power_grid.benchmark.train_table` "
            f"to train DC3 for N={N}, then pass the saved checkpoint with --checkpoint."
        )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in saved or "state_dict" not in saved:
        raise ValueError("Checkpoint must contain config and state_dict")
    cfg = saved["config"]
    if int(cfg["problem"]["N"]) != N:
        raise ValueError(f"Checkpoint was trained for N={cfg['problem']['N']}; this table requires N={N}")
    cfg["dc3"]["device"] = "cpu"
    cfg["dc3"]["corr_test_max_steps"] = MAX_ITER
    _, _, solver, dc3cfg, device, dtype = build(SPEC, cfg)
    solver.load_state_dict(saved["state_dict"])
    solver.eval()
    return solver, cfg, dc3cfg, device, dtype


def run(checkpoint: Path, instances: Path, output_dir: Path) -> None:
    x0, load, gen, case_tag = load_instances(instances)
    solver, cfg, dc3cfg, device, dtype = load_solver(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"dc3_clarabel_results_{case_tag}.csv"
    summary_path = output_dir / f"dc3_clarabel_summary_{case_tag}.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError(f"CLARABEL results for {case_tag} already exist in {output_dir}")

    reference = np.empty(x0.size, dtype=np.float64)
    for i in range(x0.size):
        _, _, reference[i], status = solve_instance(x0[i], load[i], gen[i], solver="CLARABEL", tol=1e-9)
        if status != "optimal" or not np.isfinite(reference[i]):
            raise RuntimeError(f"CLARABEL reference failed for sample {i + 1}: {status}")
        if (i + 1) % 100 == 0 or i + 1 == x0.size:
            print(f"CLARABEL: {i + 1}/{x0.size} reference instances", flush=True)

    params = to_params(x0, load, gen, device, dtype)
    scoring_problem = build_problem(cfg["problem"], torch.device("cpu"), torch.float64)
    scoring_params = to_params(x0, load, gen, torch.device("cpu"), torch.float64)

    # Prepare single-instance views before timing, like the Julia benchmark's
    # reused solver objects and already-loaded parameters.
    one = [params.index(slice(i, i + 1)) for i in range(x0.size)]
    one_scoring = [scoring_params.index(slice(i, i + 1)) for i in range(x0.size)]
    solver.solve(one[0])  # Exclude initial inference and compilation/warm-up.

    fields = ("sample", "x0", "solve_time_ms", "dc3_objective", "clarabel_J_ref",
              "opt_gap_pct", "correction_steps", "constraint_violation",
              "eq_max", "ineq_max", "domain_max")
    times = np.empty(x0.size, dtype=np.float64)
    gaps = np.empty(x0.size, dtype=np.float64)
    violations = np.empty(x0.size, dtype=np.float64)
    with csv_path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, (p, score_p) in enumerate(zip(one, one_scoring)):
            start = time.perf_counter()
            result = solver.solve(p)
            times[i] = 1000.0 * (time.perf_counter() - start)

            y = result["Y"].detach().to(device="cpu", dtype=torch.float64)
            metrics = per_instance_metrics(
                scoring_problem, score_p, y, reference[i : i + 1], dc3cfg.feas_tol
            )
            gaps[i] = float(metrics["gap_pct"][0])
            violations[i] = max(
                float(metrics["eq_max"][0]),
                float(metrics["ineq_max"][0]),
                float(metrics["domain_max"][0]),
                0.0,
            )
            writer.writerow({
                "sample": i + 1,
                "x0": x0[i],
                "solve_time_ms": times[i],
                "dc3_objective": float(metrics["obj"][0]),
                "clarabel_J_ref": reference[i],
                "opt_gap_pct": gaps[i],
                "correction_steps": int(result["steps"]),
                "constraint_violation": violations[i],
                "eq_max": float(metrics["eq_max"][0]),
                "ineq_max": float(metrics["ineq_max"][0]),
                "domain_max": float(metrics["domain_max"][0]),
            })
            file.flush()
            if (i + 1) % 100 == 0 or i + 1 == x0.size:
                print(f"DC3: {i + 1}/{x0.size} instances", flush=True)

    summary = {
        "case_tag": case_tag,
        "instances_file": str(instances),
        "checkpoint": str(checkpoint),
        "samples": int(x0.size),
        "device": str(device),
        "dtype": str(dtype),
        "max_iter": MAX_ITER,
        "reference_solver": "CLARABEL via DC3.power_grid.reference.solve_instance (tol=1e-9)",
        "gap_definition": "100 * abs(DC3 objective - CLARABEL objective) / abs(CLARABEL objective)",
        "opt_gap_mean_pct": float(gaps.mean()),
        "opt_gap_max_pct": float(gaps.max()),
        "constraint_violation_mean": float(violations.mean()),
        "constraint_violation_max": float(violations.max()),
        "solve_time_mean_ms": float(times.mean()),
        "solve_time_max_ms": float(times.max()),
        "solve_time_std_ms": float(times.std(ddof=1)) if times.size > 1 else 0.0,
        "per_instance_file": str(csv_path),
    }
    with summary_path.open("x") as file:
        json.dump(summary, file, indent=2)
    print(f"DC3 vs CLARABEL: mean opt gap = {summary['opt_gap_mean_pct']:.6g}%, "
          f"max opt gap = {summary['opt_gap_max_pct']:.6g}%")
    print(f"Constraint violation: mean = {summary['constraint_violation_mean']:.3e}, "
          f"max = {summary['constraint_violation_max']:.3e}")
    print(f"Solve time mean (max) = {summary['solve_time_mean_ms']:.3f} "
          f"({summary['solve_time_max_ms']:.3f}) ms over {x0.size} instances")
    print(f"Saved {csv_path} and {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help=f"Trained N={N} DC3 checkpoint")
    parser.add_argument("--instances", type=Path, help="Saved test inputs (default: unique N-matched file in power_table_inputs)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.checkpoint.resolve(), find_instances(args.instances), args.output_dir.resolve())


if __name__ == "__main__":
    main()
