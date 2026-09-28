"""Save fixed-forecast power-grid test inputs shared by Julia and DC3.

From the repository root, run for example::

    python3 -m DC3.power_grid.benchmark.generate_table_instances --N 96

The file contains inputs only. Julia writes IPOPT objectives to its own results
file; the DC3 benchmark solves CLARABEL references from these same inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data import read_series


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "data" / "solving_data" / "power_table_inputs"
x0_lo, x0_hi = 0.25, 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=192, help="Forecast horizon")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20262309)
    args = parser.parse_args()

    load_all, gen_all = read_series()
    if args.N < 1 or args.N > min(load_all.size, gen_all.size) or args.samples < 2:
        raise ValueError("N must fit both forecasts and samples must be at least 2")

    tag = f"N={args.N}"
    path = OUTPUT_DIR / f"test_instances_{tag}.npz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x0 = np.random.default_rng(args.seed).uniform(x0_lo, x0_hi, size=args.samples)
    load = np.broadcast_to(load_all[:args.N], (args.samples, args.N)).copy()
    gen = np.broadcast_to(gen_all[:args.N], (args.samples, args.N)).copy()
    with path.open("xb") as file:
        np.savez_compressed(file, N=np.array([args.N]), seed=np.array([args.seed]),
                            x0_bounds=np.array([x0_lo, x0_hi]),
                            x0=x0, load=load, gen=gen,
                            case_tag_utf8=np.frombuffer(tag.encode("utf-8"), dtype=np.uint8))
    print(f"Saved {args.samples} inputs (x0 ~ Uniform({x0_lo}, {x0_hi})) to {path}")


if __name__ == "__main__":
    main()