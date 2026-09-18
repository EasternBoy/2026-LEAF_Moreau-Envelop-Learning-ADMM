"""Instance generation for the economic-MPC benchmark.

`energy_mag()` in ``examples/power_grid/power_system.jl`` defines a *single*
instance: ``x0 = 0.5`` and the first ``N = 96`` samples of

    data/micro_grid/PV_48h_15-min_150kW_San_Diego.csv        (column 2, kW)
    data/micro_grid/load_15min_max100kW_SanDiego_Building.csv (column 2, kW)

DC3 is a *parametric* solver, so a family of instances is required.  The family
used here keeps the plant and the cost function untouched and varies only the
quantities that the JuMP model already exposes as ``MOI.Parameter``:

    x0           ~ U(x0_lo, x0_hi)          default [0.25, 0.75]
    (load, gen)  = the length-N window of the CSVs starting at offset s,
                   s ~ Uniform{0, ..., 192 - N}

The offset grid is split *disjointly* between train / validation / test so no
forecast window is shared across splits, and the **nominal benchmark instance**
(x0 = 0.5, s = 0) is forced into the test split as instance 0 so that the
existing reference value ``Jopt = 36479.1`` can be checked directly.

The ``x0`` range brackets the pool used for the ADMM training data in
``data_eMPC_power.jl`` (``train_pool = [1/2, 2/3, 3/4]``, ``test_pool = [3/5]``).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch

from ..common.io_utils import REPO_ROOT
from .problem import GridParams

GEN_CSV = os.path.join(REPO_ROOT, "data", "micro_grid", "PV_48h_15-min_150kW_San_Diego.csv")
LOAD_CSV = os.path.join(REPO_ROOT, "data", "micro_grid", "load_15min_max100kW_SanDiego_Building.csv")
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}


def read_series() -> tuple[np.ndarray, np.ndarray]:
    """Second column of each CSV, exactly as ``CSV.read(...)[:, 2]`` in Julia."""
    def col2(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        return np.array([float(r[1]) for r in rows[1:]], dtype=float)

    return col2(LOAD_CSV), col2(GEN_CSV)


def offset_grid(N: int, n_samples: int) -> np.ndarray:
    return np.arange(0, n_samples - N + 1, dtype=int)


def split_offsets(N: int, n_samples: int, seed: int) -> dict[str, np.ndarray]:
    """Disjoint offset pools: 60 % train, 20 % valid, 20 % test (offset 0 -> test)."""
    offs = offset_grid(N, n_samples)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(offs[offs != 0])
    n = perm.size
    n_tr = int(round(0.6 * n))
    n_va = int(round(0.2 * n))
    return {
        "train": np.sort(perm[:n_tr]),
        "valid": np.sort(perm[n_tr : n_tr + n_va]),
        "test": np.sort(np.concatenate([[0], perm[n_tr + n_va :]])),
    }


def generate_numpy(
    N: int, count: int, seed: int, split: str,
    x0_lo: float = 0.25, x0_hi: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x0, load, gen, offset)`` for `count` instances of `split`."""
    load_all, gen_all = read_series()
    n_samples = min(load_all.size, gen_all.size)
    pools = split_offsets(N, n_samples, seed)
    pool = pools[split]
    if pool.size == 0:
        raise RuntimeError(f"empty offset pool for split {split!r} (N={N}, samples={n_samples})")

    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(SPLIT_ID[split], 7))
    rng = np.random.default_rng(ss)
    offs = rng.choice(pool, size=count, replace=True)
    x0 = rng.uniform(x0_lo, x0_hi, size=count)
    if split == "test":
        offs[0], x0[0] = 0, 0.5            # nominal `energy_mag()` instance

    load = np.stack([load_all[s : s + N] for s in offs])
    gen = np.stack([gen_all[s : s + N] for s in offs])
    return x0, load, gen, offs


def to_params(x0, load, gen, device, dtype) -> GridParams:
    return GridParams(
        x0=torch.as_tensor(x0, dtype=dtype, device=device),
        load=torch.as_tensor(load, dtype=dtype, device=device),
        gen=torch.as_tensor(gen, dtype=dtype, device=device),
    )


def make_split(N, count, seed, split, device, dtype, **kw) -> GridParams:
    x0, load, gen, _ = generate_numpy(N, count, seed, split, **kw)
    return to_params(x0, load, gen, device, dtype)


def instances_path(root: str, N: int, count: int, seed: int, split: str = "test") -> str:
    return os.path.join(root, f"power_{split}_N={N}_n={count}_seed={seed}.npz")


def save_instances(path, x0, load, gen, offs, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, x0=x0, load=load, gen=gen, offset=offs,
                        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def load_instances(path):
    d = np.load(path)
    return d["x0"], d["load"], d["gen"], d["offset"]
