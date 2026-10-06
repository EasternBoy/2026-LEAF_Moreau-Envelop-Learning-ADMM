using Pkg
Pkg.activate(".")

using Printf
using SparseArrays
using JSON3
using LinearAlgebra
using StaticArrays
using NPZ
using Pkg
using Plots
using Base.Threads
using Random, Distributions

import ParametricOptInterface as POI
import MathOptInterface       as MOI

const FloatType = Float64
const n::Int = 100
const neq::Int = 50
const m::Int = 50

include("../../problems/qp/problem.jl")
include("../../problems/qp/setup.jl")
include("../../problems/qp/jump_solver.jl")
include("../../problems/qp/admm.jl")

n_train = 20
n_test  = 10



para_opt = data_opt(n, neq, m)


solver_name = "osqp"
data_train  = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])

# Generate solvers
admm_sol    = ADMM_eco_iter_data(para_opt, solver_name; tol=1e-6, max_iter=1000)

opt_ADMM = zeros(FloatType, para_opt.n)

mkpath(joinpath("data", "qp", "training"))

data_train  = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])
Random.seed!(1)
data_gen!((rand(Uniform(-6, 6), n) for _ in 1:20000), data_train)

@printf("Collected %4d training data points \n\n", length(data_train["input"]))
data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => para_opt.rho, "enve" => data_train["env"],
            "Q" => para_opt.Q, "p" => para_opt.p, "A" => para_opt.A, "G" => para_opt.G, "h" => para_opt.h)
npzwrite(joinpath("data", "qp", "training", string("qp-manual-","rho=",para_opt.rho,"-train",".npz")), data)




# Testing dataset
Random.seed!(2)
data_test  = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])
for _ in 1:n_test
    new_para = data_opt(n, neq, m)
    println("================ Collecting testing data ================")
    opt_ADMM[:], J_ADMM =  admm_sol(data_test, new_para; verbose=true)
    _, _, J_opt = JuMP_solver(solver_name, new_para, 1e-9)
    gap = 100abs(J_ADMM - J_opt)/max(abs(J_opt), eps(FloatType))
    eq_viol = maximum(abs, new_para.A*opt_ADMM - new_para.x)
    ineq_viol = max(maximum(new_para.G*opt_ADMM - new_para.h), 0.0)
    @printf("J_ADMM = %.8f, OSQP gap = %.6g%%, eq = %.3e, ineq = %.3e\n", J_ADMM, gap, eq_viol, ineq_viol)
end
@printf("Collected %4d testing data points \n\n", length(data_test["input"]))
data = Dict("input" => reduce(hcat, data_test["input"]), "grad" => reduce(hcat, data_test["grad"]), "rho" => para_opt.rho, "enve" => data_test["env"],
            "Q" => para_opt.Q, "p" => para_opt.p, "A" => para_opt.A, "G" => para_opt.G, "h" => para_opt.h)
npzwrite(joinpath("data", "qp", "training", string("qp-manual-","rho=",para_opt.rho,"-test",".npz")), data)
