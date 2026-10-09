using JuMP
import OSQP
import Clarabel

using Distributions
using LinearAlgebra
using SparseArrays
using JSON3

# min ½y'Qy + p'y + λ‖y‖₁  s.t.  Ay = x, Gy ≤ h, in the original decision-variable coordinates.
var_scale(data) = 1.0

const QP_L1_LAMBDA = 1.0
const QP_L1_RHO    = 10.0   # ADMM / Moreau-envelope ρ: the fewest exact sMEL-ADMM iterations at λ = 1

# p, A, G, h: locuslab/DC3/datasets/simple/make_dataset.py (NumPy seed 17, original draw order),
# as in problems/qp.  Q is dense: B B'/400 + 0.1 I with B ~ N(0, 1), 100 × 100, NumPy seed 18,
# so its eigenvalues lie in about [0.1, 1.1].  DC3/qp_l1/data.py draws the same matrices.
function generate_qp_l1_data()
    script = """
import json
import numpy as np
np.random.seed(17)
np.random.random(100)
p = np.random.random(100)
A = np.random.normal(size=(50, 100))
X = np.random.uniform(-1, 1, size=(10000, 50))
G = np.random.normal(size=(50, 100))
h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
B = np.random.RandomState(18).normal(size=(100, 100))
Q = B @ B.T / 400 + 0.1 * np.eye(100)
print(json.dumps(dict(Q=Q.flatten(order='F').tolist(), p=p.tolist(),
                      A=A.flatten(order='F').tolist(),
                      G=G.flatten(order='F').tolist(), h=h.tolist())))
"""
    data = JSON3.read(read(`python3 -c $script`, String), Dict{String, Any})
    Q = reshape(Float64.(data["Q"]), 100, 100)
    return Dict("Q" => (Q + Q')/2, "p" => Float64.(data["p"]),
                "A" => reshape(Float64.(data["A"]), 50, 100),
                "G" => reshape(Float64.(data["G"]), 50, 100),
                "h" => Float64.(data["h"]), "lambda" => fill(QP_L1_LAMBDA), "rho" => fill(QP_L1_RHO))
end

# The matrices are fixed across instances; only x varies.
const qp_data = generate_qp_l1_data()

@kwdef struct data_opt
    n::Int = length(qp_data["p"])
    neq::Int = size(qp_data["A"], 1)
    nineq::Int = size(qp_data["G"], 1)
    Q::Matrix{Float64} = qp_data["Q"]
    p::Vector{Float64} = qp_data["p"]
    A::Matrix{Float64} = qp_data["A"]
    G::Matrix{Float64} = qp_data["G"]
    h::Vector{Float64} = qp_data["h"]
    λ::Float64 = qp_data["lambda"][]
    x::Vector{Float64} = rand(Uniform(-1, 1), neq)
    rho::Float64 = qp_data["rho"][]
end

function data_opt(n::Int, neq::Int, nineq::Int)
    @assert n == length(qp_data["p"])
    @assert (neq, n) == size(qp_data["A"])
    @assert (nineq, n) == size(qp_data["G"])
    return data_opt(; n, neq, nineq)
end

@inbounds function get_objective(data::data_opt, w::Vector{FloatType})
    return 0.5 * dot(w, data.Q * w) + dot(data.p, w) + data.λ * sum(abs, w)
end

const QP_L1_EIGS = extrema(eigvals(Symmetric(qp_data["Q"])))

soft_threshold(u, κ) = sign(u) * max(abs(u) - κ, 0.0)

"""
prox of f = ½x'Qx + p'x + λ‖x‖₁ with parameter ρ, and the Moreau envelope
f(prox) + ρ/2‖prox - q‖², by accelerated proximal gradient (FISTA with the constant
momentum of the μ-strongly convex case) on ½x'(Q + ρI)x + (p - ρq)'x + λ‖x‖₁.
Stops when an iteration moves no coordinate by more than `tol`.
"""
function prox_env(q::Vector{FloatType}; ρ = qp_data["rho"][], λ = QP_L1_LAMBDA,
                  tol = 1e-13, max_iter = 10_000)
    Q, p = qp_data["Q"], qp_data["p"]
    μ, L = QP_L1_EIGS[1] + ρ, QP_L1_EIGS[2] + ρ
    momentum = (sqrt(L) - sqrt(μ)) / (sqrt(L) + sqrt(μ))
    x = copy(q); y = copy(q); x_prev = similar(q); g = similar(q)
    for _ in 1:max_iter
        copyto!(x_prev, x)
        mul!(g, Q, y)
        @. g += p + ρ * (y - q)
        @. x = soft_threshold(y - g / L, λ / L)
        @. y = x + momentum * (x - x_prev)
        maximum(abs, x - x_prev) <= tol && break
    end
    env = 0.5 * dot(x, Q * x) + dot(p, x) + λ * sum(abs, x) + ρ / 2 * sum(abs2, x - q)
    return x, env
end
