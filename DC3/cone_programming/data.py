"""Instance generation for the maximum-entropy cone program.

The generating distribution is copied from the two-argument constructor
``data_opt(n, m)`` in ``examples/cone_programming/maxEntropy.jl``::

    A = rand(Uniform(0,1), m, n)
    b = [sum(A[i,:]) / (1.06 n) for i in 1:m]

**Documented deviation.**  The draws come from NumPy's PCG64 rather than
Julia's Xoshiro, because the instances have to be shared between the Python DC3
code, the Python (cvxpy) reference solver and the Julia (Ipopt / sLME-ADMM)
baselines.  The *distribution* is identical; the particular realisations are
not the ones a Julia run would produce.

Train / validation / test splits use disjoint `SeedSequence` spawn keys, so the
three sets are guaranteed disjoint and each is reproducible from the base seed
alone.  Only the *test* split is written to disk (an (N, m, n) array of
training instances would be hundreds of MB); train/valid are regenerated
deterministically on demand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .problem import ConeParams

SPLIT_ID = {"train": 0, "valid": 1, "test": 2}
B_DIVISOR = 1.06          # `b = [sum(A[i,:])/(1.06*n) ...]`


def generate_numpy(n: int, m: int, count: int, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (A, b) for `count` instances of the given split."""
    if split not in SPLIT_ID:
        raise ValueError(f"unknown split {split!r}")
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(SPLIT_ID[split],))
    rng = np.random.default_rng(ss)
    A = rng.uniform(0.0, 1.0, size=(count, m, n))
    b = A.sum(axis=2) / (B_DIVISOR * n)
    return A, b


def to_params(A: np.ndarray, b: np.ndarray, device, dtype) -> ConeParams:
    return ConeParams(
        A=torch.as_tensor(A, dtype=dtype, device=device),
        b=torch.as_tensor(b, dtype=dtype, device=device),
    )


def make_split(n, m, count, seed, split, device, dtype) -> ConeParams:
    A, b = generate_numpy(n, m, count, seed, split)
    return to_params(A, b, device, dtype)


def test_set_path(root: str, n: int, m: int, count: int, seed: int) -> str:
    return os.path.join(root, f"cone_test_n={n}_m={m}_N={count}_seed={seed}.npz")


def save_test_set(path: str, A: np.ndarray, b: np.ndarray, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, A=A, b=b, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def load_test_set(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["A"], d["b"]
