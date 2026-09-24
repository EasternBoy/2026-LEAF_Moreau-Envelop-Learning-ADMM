"""Maximum-entropy cone program, transcribed from ``examples/entr_max``.

Source of truth
---------------
``examples/entr_max/maxEntropy.jl`` + ``JuMPsolver.jl`` build, with
``scale = 2n``::

    variable  x_i >= 1e-8
    s.t.      sum(x) == scale
              A x <= b * scale
    minimise  sum_i x_i (log x_i - log scale)

and report ``x/scale`` and ``objective_value/scale``.  Substituting
``w = x / scale`` this is *exactly*

    min_w   sum_i w_i log w_i                         (negative entropy)
    s.t.    1^T w = 1                                 (n_eq = 1)
            A w <= b                                  (m rows)
            w_i >= 1e-8 / scale                       (n rows)

with instance parameters ``A ~ U(0,1)^{m x n}`` and
``b_i = sum_j A_ij / (1.06 n)`` (the two-argument ``data_opt(n, m)``
constructor used by every benchmark script; the keyword default in the struct
uses 1.1 instead of 1.06 and is *not* the one exercised).

DC3 structure
-------------
``n_eq = 1``, so the partial block has ``n - 1`` variables and the completion is

    w_D = 1 - sum_{i in P} w_i,

i.e. ``A_D = [1]`` is trivially invertible (``cond = 1``) for *any* choice of the
single dependent index.  The default is the last coordinate, which is what
column-pivoted QR also returns for ``A_eq = 1^T`` up to tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..common.problem import ParametricProblem


@dataclass
class ConeParams:
    """A batch of instances.  ``A``: (B, m, n); ``b``: (B, m)."""

    A: torch.Tensor
    b: torch.Tensor

    def __len__(self) -> int:
        return self.A.shape[0]

    def index(self, idx) -> "ConeParams":
        return ConeParams(A=self.A[idx], b=self.b[idx])

    def to(self, device, dtype) -> "ConeParams":
        return ConeParams(A=self.A.to(device=device, dtype=dtype),
                          b=self.b.to(device=device, dtype=dtype))


def index_cone(params: ConeParams, idx) -> ConeParams:
    return params.index(idx)


class MaxEntropyProblem(ParametricProblem):
    name = "max_entropy"

    def __init__(
        self,
        n: int,
        m: int,
        feature_mode: str = "A_flat_b",
        w_floor: Optional[float] = None,
        w_margin: float = 0.0,
        ineq_margin: float = 0.0,
        log_eps: float = 1e-30,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ):
        self.n, self.m = int(n), int(m)
        self.scale = 2 * self.n                      # `const scale::Int = 2n`
        # `@variable(model, x[1:n] >= 1e-8)` with x = scale * w
        self.w_floor = (1e-8 / self.scale) if w_floor is None else float(w_floor)
        self.w_margin = float(w_margin)
        self.ineq_margin = float(ineq_margin)
        self.log_eps = float(log_eps)
        self.feature_mode = feature_mode
        self.dtype, self.device = dtype, device

        self.n_y = self.n
        self.n_eq = 1
        self.n_ineq = self.m + self.n
        self.A_eq = torch.ones(1, self.n, dtype=dtype, device=device)
        self.x_dim = {
            "A_flat": self.m * self.n,
            "A_flat_b": self.m * self.n + self.m,
            "b": self.m,
        }[feature_mode]

    # -- instance plumbing -------------------------------------------------
    def features(self, p: ConeParams) -> torch.Tensor:
        if self.feature_mode == "A_flat":
            return p.A.reshape(p.A.shape[0], -1)
        if self.feature_mode == "A_flat_b":
            return torch.cat([p.A.reshape(p.A.shape[0], -1), p.b], dim=1)
        return p.b

    def eq_rhs(self, p: ConeParams) -> torch.Tensor:
        return torch.ones(len(p), 1, dtype=self.dtype, device=p.A.device)

    # -- objective ---------------------------------------------------------
    def obj_fn(self, p: ConeParams, Y: torch.Tensor, safe: bool = True) -> torch.Tensor:
        """``sum_i w_i log w_i`` (the Julia ``objective_value/scale``).

        ``0 log 0`` is taken as 0 (the continuous extension), matching the
        convention of the entropy cone used by Clarabel/ECOS.  With ``safe`` the
        argument of the log is clamped at ``log_eps`` so that negative iterates
        produce a large-but-finite value rather than ``nan``.
        """
        if safe:
            W = torch.clamp(Y, min=self.log_eps)
            return (Y * torch.log(W)).sum(dim=1)
        return torch.special.xlogy(Y, Y).sum(dim=1)  # continuous extension: 0 log 0 = 0

    # -- constraints -------------------------------------------------------
    def ineq_resid(self, p: ConeParams, Y: torch.Tensor, margin: bool = False) -> torch.Tensor:
        g1 = torch.bmm(p.A, Y.unsqueeze(2)).squeeze(2) - p.b       # A w - b <= 0
        floor = self.w_floor + (self.w_margin if margin else 0.0)
        g2 = floor - Y                                              # floor - w <= 0
        if margin and self.ineq_margin:
            g1 = g1 + self.ineq_margin
        return torch.cat([g1, g2], dim=1)

    def domain_resid(self, p: ConeParams, Y: torch.Tensor) -> torch.Tensor:
        """``w >= 0`` is needed for ``w log w``; returns ``-w`` (positive = bad)."""
        return -Y

    # -- closed-form correction gradient ----------------------------------
    def ineq_partial_grad(self, p: ConeParams, Z: torch.Tensor, completion,
                          row_scale=None) -> torch.Tensor:
        """d/dZ || relu(g) ||^2 with g affine in Y and Y affine in Z.

        With ``M = A_D^{-1} A_P`` (here a row of ones) the reduced constraint
        Jacobian is ``G_eff = G_P - G_D M`` and the gradient is
        ``2 * relu(g)^T G_eff = 2 * ( r^T G_P - (r^T G_D) M )``.
        """
        b_eq = self.eq_rhs(p)
        Y = completion.complete(Z, b_eq)
        r = self.ineq_dist(p, Y, margin=True)                       # (B, m+n)
        if row_scale is not None:
            r = r * (row_scale ** 2)
        P, D, M = completion.partial_vars, completion.other_vars, completion.A_other_inv_A_partial

        r1, r2 = r[:, : self.m], r[:, self.m :]
        # contribution of  A w - b  :  G = A
        rA = torch.bmm(r1.unsqueeze(1), p.A).squeeze(1)             # (B, n)
        t1 = rA[:, P] - rA[:, D] @ M
        # contribution of  floor - w :  G = -I
        t2 = -r2[:, P] + r2[:, D] @ M
        return 2.0 * (t1 + t2)

    def partial_bounds(self, completion):
        """`w_i > 0` is required by `w log w`; there is no useful upper bound."""
        lo = torch.full((completion.n_partial,), max(self.w_floor, 1e-12), dtype=torch.float64)
        hi = torch.full((completion.n_partial,), float("inf"), dtype=torch.float64)
        return lo, hi

    def partial_init_target(self, completion):
        """Start at the uniform distribution `w_i = 1/n` (the unconstrained optimum)."""
        return torch.full((completion.n_partial,), 1.0 / self.n, dtype=torch.float64)

    def unpack(self, Y: torch.Tensor) -> dict:
        return {"w": Y}

    def to(self, device, dtype):
        self.A_eq = self.A_eq.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        return self
