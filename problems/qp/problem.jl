using JuMP
import OSQP

using Distributions
using LinearAlgebra
using SparseArrays
using JSON3

# The QP model is trained in the original decision-variable coordinates.
var_scale(data) = 1.0

# Reproduce locuslab/DC3/datasets/simple/make_dataset.py exactly.
# NumPy's RNG and draw order are required to reproduce the paper's matrices.
function generate_dc3_data()
    script = """
import json
import numpy as np
np.random.seed(17)
Q = np.random.random(100)
p = np.random.random(100)
A = np.random.normal(size=(50, 100))
X = np.random.uniform(-1, 1, size=(10000, 50))
G = np.random.normal(size=(50, 100))
h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
print(json.dumps(dict(Q=Q.tolist(), p=p.tolist(),
                      A=A.flatten(order='F').tolist(),
                      G=G.flatten(order='F').tolist(), h=h.tolist())))
"""
    data = JSON3.read(read(`python3 -c $script`, String), Dict{String, Any})
    return Dict("Q" => Matrix(Diagonal(Float64.(data["Q"]))),
                "p" => Float64.(data["p"]),
                "A" => reshape(Float64.(data["A"]), 50, 100),
                "G" => reshape(Float64.(data["G"]), 50, 100),
                "h" => Float64.(data["h"]), "rho" => fill(1.0))
end

# The matrices are fixed across instances; only x varies.
const qp_data = generate_dc3_data()

@kwdef struct data_opt
    n::Int = length(qp_data["p"])
    neq::Int = size(qp_data["A"], 1)
    nineq::Int = size(qp_data["G"], 1)
    Q::Matrix{Float64} = qp_data["Q"]
    p::Vector{Float64} = qp_data["p"]
    A::Matrix{Float64} = qp_data["A"]
    G::Matrix{Float64} = qp_data["G"]
    h::Vector{Float64} = qp_data["h"]
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
    return 0.5 * dot(w, data.Q * w) + dot(data.p, w)
end
