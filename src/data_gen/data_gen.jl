using Pkg

Pkg.activate(".")
# Pkg.instantiate()

using LinearAlgebra, NPZ
using Printf


include("system.jl")
include("../MPC_baseline/MPC_JuMPsolver.jl")
include("../ADMM_baseline/MPC_ADMM.jl")

get_para = LTI_2D
mpc_para, cost_func = get_para() 
tol = 1e-4

data_train = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])

train_pool = [[3., 1.], [1., 4.], [2., 2.], [1., 1.]]
test_pool  = [[3., 2.], [2., 3.]]

if typeof(cost_func) == quadratic || typeof(cost_func) == L1norm
    solver_name = "Gurobi"
else solver_name = "Ipopt" end

mpc_spa_solver = MPC_solver_gen(solver_name, mpc_para)
mpc_den_solver = MPCDense_solver_gen(solver_name, mpc_para)
prime_model    = prime_solver("madNLP", mpc_para)
aux_model      = aux_solver("Gurobi", mpc_para)
admm_sol       = ADMM_iter(mpc_para, prime_model, aux_model; tol = tol)


#Training dataset
for x0 in train_pool
    println("================ Train x0 =",round.(x0, digits = 3),"================")
    println("================ $solver_name ================")
    u_spa, J_spa, solve_time = mpc_spa_solver(x0)
    u_den, _ ,_ = mpc_den_solver(x0)
    @printf("J_opt = %5.3f, solving time = %5.2f ms\n", J_spa, solve_time*1000) 
    @printf("||u_den - u_spa|| = %5.3f\n", norm(u_spa - u_den))

    println("================ADMM ================")
    u_ADMM, J_ADMM ,_ =  admm_sol(x0; data  = data_train, verbose = true)
    # @printf("J_ADMM = %5.3f\n", J_ADMM)
    @printf("ΔJ/J = %5.2f%%\n\n", abs((J_spa - J_ADMM)*100/J_spa))
end
@printf("Collected %4d data points \n\n", length(data_train["input"]))
data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => mpc_para.rho, "enve" => data_train["env"])
npzwrite(joinpath("data", string(typeof(cost_func),"-rho=", mpc_para.rho,"-",get_para,"-train-cstr",".npz")), data)


# Collectors
data_test = Dict("input" => Vector{Float64}[], "env" => Float64[], "grad" => Vector{Float64}[])

# Make testing dataset
for x0 in test_pool
    println("================ Test: x0 = $x0 ================")
    u_ADMM, J_ADMM ,_ =  admm_sol(x0; data  = data_test, verbose = true)
    println()
end
@printf("Collected %4d data points \n\n", length(data_test["input"]))
data = Dict("input" => reduce(hcat, data_test["input"]), "grad" => reduce(hcat, data_test["grad"]), "rho" => mpc_para.rho, "enve" => data_test["env"])
npzwrite(joinpath("data", string(typeof(cost_func),"-rho=", mpc_para.rho,"-",get_para,"-test-cstr",".npz")), data)