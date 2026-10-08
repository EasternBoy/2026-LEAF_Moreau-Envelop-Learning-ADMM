"""min 0.5 y'Qy + p'y + lam ||y||_1 subject to Ay=x, Gy<=h, matching problems/qp_l1.

DC3 works on y directly: autograd uses sign(y) as the subgradient of ||y||_1, and the
correction only follows the inequality violations, which do not involve the L1 term."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..common.problem import ParametricProblem


@dataclass
class QPParams:
    """A batch of equality right-hand sides, x: (B, neq)."""

    x: torch.Tensor

    def __len__(self):
        return self.x.shape[0]

    def index(self, idx):
        return QPParams(x=self.x[idx])

    def to(self, device, dtype):
        return QPParams(x=self.x.to(device=device, dtype=dtype))


def index_qp(params: QPParams, idx) -> QPParams:
    return params.index(idx)


class QPProblem(ParametricProblem):
    name = "qp_l1"

    def __init__(self, Q, p, A, G, h, lam, device, dtype):
        self.lam = float(lam)
        self.Q = torch.as_tensor(Q, dtype=dtype, device=device)
        self.p = torch.as_tensor(p, dtype=dtype, device=device)
        self.A_eq = torch.as_tensor(A, dtype=dtype, device=device)
        self.G = torch.as_tensor(G, dtype=dtype, device=device)
        self.h = torch.as_tensor(h, dtype=dtype, device=device)
        self.n_y = self.Q.shape[0]
        self.n_eq = self.A_eq.shape[0]
        self.n_ineq = self.G.shape[0]
        self.x_dim = self.n_eq

    def features(self, params):
        return params.x

    def eq_rhs(self, params):
        return params.x

    def obj_fn(self, params, Y, safe=True):
        return 0.5 * (Y * (Y @ self.Q)).sum(dim=1) + Y @ self.p + self.lam * Y.abs().sum(dim=1)

    def ineq_resid(self, params, Y, margin=False):
        return Y @ self.G.T - self.h

    def ineq_partial_grad(self, params, Z, completion, row_scale=None):
        Y = completion.complete(Z, self.eq_rhs(params))
        r = self.ineq_dist(params, Y, margin=True)
        if row_scale is not None:
            r = r * row_scale ** 2
        P, D = completion.partial_vars, completion.other_vars
        G_eff = self.G[:, P] - self.G[:, D] @ completion.A_other_inv_A_partial
        return 2.0 * r @ G_eff

    def to(self, device, dtype):
        for key in ("Q", "p", "A_eq", "G", "h"):
            setattr(self, key, getattr(self, key).to(device=device, dtype=dtype))
        return self
