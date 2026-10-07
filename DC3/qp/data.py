"""Fixed QP matrices matching problems/qp/problem.jl and shared Julia test inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .problem import QPParams

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ID = {"train": 0, "valid": 1}


def fixed_data() -> dict:
    # Preserve NumPy's original DC3 seed and draw order, including the X draw.
    rng = np.random.RandomState(17)
    Q = np.diag(rng.random_sample(100))
    p = rng.random_sample(100)
    A = rng.normal(size=(50, 100))
    rng.uniform(-1, 1, size=(10000, 50))
    G = rng.normal(size=(50, 100))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
    return dict(Q=Q, p=p, A=A, G=G, h=h)


def to_params(X, device, dtype) -> QPParams:
    return QPParams(x=torch.as_tensor(X, dtype=dtype, device=device))


def make_split(count, seed, split, test_instances, device, dtype) -> QPParams:
    if split == "test":
        path = Path(test_instances)
        if not path.is_absolute():
            path = REPO_ROOT / path
        with np.load(path) as saved:
            matrices = fixed_data()
            for key, expected in matrices.items():
                if not np.allclose(saved[key], expected, rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{path}: {key} differs from the Julia QP formulation")
            X = saved["X"]
            if X.ndim != 2 or X.shape[1] != matrices["A"].shape[0] or count > len(X):
                raise ValueError(f"{path}: incompatible test dimensions/count")
            X = X[:count].copy()
    else:
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(SPLIT_ID[split],)))
        X = rng.uniform(-1, 1, size=(count, 50))
    return to_params(X, device, dtype)
