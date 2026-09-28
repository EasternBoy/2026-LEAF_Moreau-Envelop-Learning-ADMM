using Pkg
Pkg.activate(".")

Pkg.instantiate()

using NPZ, Printf
using QuasiMonteCarlo
using Random
using Distributions
using LinearAlgebra
using Plots
using JuMP 
using Clarabel
using PyPlot

include("../../problems/mvee/problem.jl")
include("../../problems/mvee/jump_solver.jl")
include("../../problems/mvee/admm.jl")
include("../../problems/mvee/utils.jl")

opt_GT   = nothing
opt_ADMM = nothing

mpc_para  = data_opt()
cost_func = mpc_para.cost_func


solver_name = "Clarabel"

data_train  = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[], "org_f" => FloatType[])
data_test   = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[], "org_f" => FloatType[])


solver_GT   = logdet_solver(solver_name, mpc_para)

prime_sol   = prime_solver_data("Clarabel", mpc_para)
aux_sol     = aux_solver_data("Clarabel", mpc_para)
admm_sol    = ADMM_logdet_iter(mpc_para, prime_sol, aux_sol)

n = mpc_para.n
m = mpc_para.m
A_para = mpc_para.A

train_pool = []
test_pool  = []

for _ in 1:300
    A_para = random_A_rotated(n, m)
    push!(train_pool, A_para)
end 

for _ in 1:20
    A_para = random_A_rotated(n, m)
    push!(test_pool, A_para)
end

println("================ TRAIN ================")
for A_para in train_pool
    global opt_GT, opt_ADMM
    mpc_para.A = A_para
    # println("----------------- $A_para -----------------")
    println("---------------- $solver_name ----------------")
    opt_GT, _, obj_GT = solver_GT(mpc_para)
    println("objective function = ", obj_GT)

    println("---------------- ADMM ----------------")
    opt_ADMM, obj_ADMM = admm_sol(data_train, A_para; verbose=true)
    println("objective function = ", obj_ADMM)
    println("")

    if abs(obj_GT - obj_ADMM) > 0.1
        println("Warning: Large optimality gap")
        println("optimal solution = ", opt_GT)
        println("ADMM solution = ", opt_ADMM)
    end
    # plot_ellipsoid(opt_ADMM, opt_GT, mpc_para.A, obj_GT, obj_ADMM; x_scale=1., y_scale=1.)

end
@printf("Collected %4d training data points \n\n", length(data_train["input"]))
data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => mpc_para.rho, "enve" => data_train["env"], "org_f" => data_train["org_f"])
npzwrite(joinpath("data", "mvee", "training", string("logdet-rho=", mpc_para.rho,"-train","-m=", mpc_para.m,".npz")), data)

println("================ TEST ================")
# Testing dataset
for A_para in test_pool
    mpc_para.A = A_para
        opt_GT,_, obj_GT = solver_GT(mpc_para)
    if maximum(abs.(opt_GT)) > 50
        println("Skipped: GT solution too large (max entry = $(maximum(abs.(opt_GT))) )")
        continue   # skip this sample
    end
    opt_ADMM, obj_ADMM = admm_sol(data_test, A_para; verbose = true)
end
@printf("Collected %4d testing data points \n\n", length(data_test["input"]))
data = Dict("input" => reduce(hcat, data_test["input"]), "grad" => reduce(hcat, data_test["grad"]), "rho" => mpc_para.rho, "enve" => data_test["env"], "org_f" => data_test["org_f"])
npzwrite(joinpath("data", "mvee", "training", string("logdet-rho=", mpc_para.rho,"-test", "-m=", mpc_para.m,".npz")), data)


