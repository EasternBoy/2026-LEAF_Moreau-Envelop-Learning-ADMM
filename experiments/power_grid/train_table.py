"""Train DC3 for the fixed-forecast power-grid table instances.

Run from the repository root::

    python3 experiments/power_grid/generate_table_instances.py --N 96
    python3 experiments/power_grid/train_table.py --tag table-N96

The resulting checkpoint is printed at the end. By default it is saved to
DC3/results/power_grid-table-N96/checkpoint.pt and can be passed directly to
``experiments/power_grid/table_benchmark.py``. The horizon comes from
``table_benchmark.N``.
Training uses the forecasts in the saved test-input file,
with independent x0 draws; the saved test x0 values remain held out.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from DC3.common.io_utils import DC3_ROOT  # noqa: E402
from DC3.common.runner import add_common_args, apply_overrides, load_config, run_training  # noqa: E402
from DC3.power_grid.data import to_params  # noqa: E402
from DC3.power_grid.experiment import SPEC  # noqa: E402
from generate_table_instances import X0_HI, X0_LO  # noqa: E402
from table_benchmark import N, find_instances, load_instances  # noqa: E402


def fixed_forecast_split(dc: dict, split: str, count: int, device, dtype, *, load, gen):
    """Independent x0 draws with the forecasts from the saved test file."""
    if split not in ("train", "valid") or count < 1:
        raise ValueError("Expected a nonempty train or validation split")
    seed = np.random.SeedSequence([int(dc["seed"]), N, 0 if split == "train" else 1])
    x0 = np.random.default_rng(seed).uniform(X0_LO, X0_HI, size=count)
    return to_params(x0, np.repeat(load[None, :], count, axis=0),
                     np.repeat(gen[None, :], count, axis=0), device, dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--instances", type=Path,
                        help="Saved test-input NPZ (default: the unique N-matched file)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    instances = find_instances(args.instances)
    x0_test, loads, gens, _ = load_instances(instances)
    if not np.all(loads == loads[0]) or not np.all(gens == gens[0]):
        raise ValueError("This trainer requires identical load and generation forecasts across test instances")
    if np.any((x0_test < X0_LO) | (x0_test > X0_HI)):
        raise ValueError(f"Test x0 values must lie in [{X0_LO}, {X0_HI}]")
    spec = replace(SPEC, build_split=partial(fixed_forecast_split, load=loads[0], gen=gens[0]))

    cfg = apply_overrides(load_config(SPEC, args.config), args.set)
    cfg["problem"]["N"] = N
    cfg["data"]["N"] = N
    cfg["data"]["x0_lo"] = X0_LO
    cfg["data"]["x0_hi"] = X0_HI
    cfg["data"]["table_instances_file"] = str(instances)
    cfg["dc3"]["device"] = "cpu"
    cfg["dc3"]["dtype"] = "float64"
    tag = args.tag or f"table-N{N}"
    checkpoint = Path(DC3_ROOT) / "results" / f"power_grid-{tag}" / "checkpoint.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}. Choose another --tag.")

    print(f"Training DC3 at N={N} with forecasts from {instances} and independent x0 ~ Uniform({X0_LO}, {X0_HI})")
    result = run_training(spec, cfg, tag=tag, quiet=args.quiet)
    print("Run the table benchmark with:")
    print(f"  python3 experiments/power_grid/table_benchmark.py --checkpoint {result['checkpoint']}")


if __name__ == "__main__":
    main()
