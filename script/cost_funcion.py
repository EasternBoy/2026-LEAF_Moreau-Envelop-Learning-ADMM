import cvxpy as cp
import numpy as np


def logsum(x, u, Q=None, R=None):
    """
    log(sum(exp(x))) + ||u||²
    """
    return cp.log_sum_exp(x) + cp.sum_squares(u)


def quadratic(x, u, Q, R):
    """
    Quadratic cost:
    xᵀQx + uᵀRu
    """
    return cp.quad_form(x, Q) + cp.quad_form(u, R)


# def L1norm_total(x, u, Q, R, N):
#     """
#     L1-norm cost over full horizon:
#     sum_k ( ||Q x_k||₁ + ||R u_k||₁ )
#     """
#     cost = 0
#     for k in range(N):
#         cost += cp.norm1(Q @ x[:, k]) + cp.norm1(R @ u[:, k])
#     return cost


def L1norm_single(x, u, Q, R):
    """
    L1-norm cost for single step:
    ||Qx||₁ + ||Ru||₁
    """
    return cp.norm1(Q @ x) + cp.norm1(R @ u)

def nonSmooth(x, u, Q=None, R=None, δ=1.0):
    """
    Nonsmooth Huber-like cost:
    if |u| < δ → ||u||²
    else → δ(|u| - δ/2)
    plus ||x||²
    """
    # Huber loss elementwise
    huber_u = cp.sum(cp.huber(u, M=δ))
    return huber_u + cp.sum_squares(x)