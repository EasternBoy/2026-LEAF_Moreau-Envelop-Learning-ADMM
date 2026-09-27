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

import ParametricOptInterface as POI
import MathOptInterface       as MOI

const FloatType = Float64
const n::Int = 1000
const m::Int = 100

include("../../problems/entr_max/problem.jl")
include("../../problems/entr_max/setup.jl")
include("../../problems/entr_max/jump_solver.jl")
include("../../problems/entr_max/admm.jl")

n_train = 20
n_test  = 10



para_opt = data_opt(n, m)


cost_func   = para_opt.cost_func        
solver_name = "Ipopt"
data_train  = Dict("input" => FloatType[], "env" => FloatType[], "grad" => FloatType[])

# Generate solvers
admm_sol    = ADMM_eco_iter_data(para_opt, solver_name; tol=1e0)

# opt_sol  = zeros(FloatType, para_opt.n)
opt_ADMM = zeros(FloatType, para_opt.n)

# # Training dataset
# for _ in 1:n_train
#     new_para = data_opt(n, m)
#     println("================ Collecting training data ================")
#     println("---------------- $solver_name ----------------")
#     opt_sol[:], solve_time, cost_opt = JuMP_solver(solver_name, new_para, 1e-6)
#     @printf("J_opt = %5.3f, solving time = %5.2f ms\n", cost_opt, solve_time*1000) 
    

#     println("---------------- ADMM ----------------")
#     opt_ADMM[:], J_ADMM =  admm_sol(data_train, new_para)
#     @printf("J_ADMM = %5.3f\n", J_ADMM) 
#     @printf("ΔJ/J = %5.3f%%\n", abs(cost_opt - J_ADMM) / abs(cost_opt) * 100)
#     @printf("max|opt_sol  - opt_ADMM| = %5.3f\n\n", maximum(abs.(opt_ADMM - opt_sol)))
# end


# @printf("Collected %4d training data points \n\n", length(data_train["input"]))
# data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => para_opt.rho, "enve" => data_train["env"])
# npzwrite(joinpath("data", "entr_max", "training", string("maxEntropy-n=", n, "m=", m, "rho=",para_opt.rho,"-train",".npz")), data)


data_train  = Dict("input" => FloatType[], "env" => FloatType[], "grad" => FloatType[])
data_gen!(range(-50, 50, length = 20000), data_train)

@printf("Collected %4d training data points \n\n", length(data_train["input"]))
data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => para_opt.rho, "enve" => data_train["env"])
npzwrite(joinpath("data", "entr_max", "training", string("maxEntropy-manual-","rho=",para_opt.rho,"-train",".npz")), data)




# Testing dataset
data_test  = Dict("input" => FloatType[], "env" => FloatType[], "grad" => FloatType[])
for _ in 1:n_test
    new_para = data_opt(n, m)
    println("================ Collecting testing data ================")
    opt_ADMM[:], J_ADMM =  admm_sol(data_test, new_para)
    @printf("J_ADMM = %5.3f\n", J_ADMM) 
end
@printf("Collected %4d testing data points \n\n", length(data_test["input"]))
data = Dict("input" => reduce(hcat, data_test["input"]), "grad" => reduce(hcat, data_test["grad"]), "rho" => para_opt.rho, "enve" => data_test["env"])
npzwrite(joinpath("data", "entr_max", "training", string("maxEntropy-n=", n, "m=", m, "rho=",para_opt.rho,"-test",".npz")), data)