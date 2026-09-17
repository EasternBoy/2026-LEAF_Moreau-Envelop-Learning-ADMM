using Pkg
Pkg.activate(".")
Pkg.instantiate()

using NPZ, Printf

include("system.jl")
include("../MPC_baseline/MPC_JuMPsolver.jl")
include("../ADMM_baseline/eMPC_ADMM.jl")

get_para = power_share_star
mpc_para, cost_func = get_para()
tol = 1e-4
                                        
if typeof(cost_func) == quadratic || typeof(cost_func) == L1norm
    solver_name = "Gurobi"
else solver_name = "Ipopt" end

data_train = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])


mpc_eco_sol = mpc_eco_solver(solver_name, mpc_para)
prime_sol   = prime_solver_eco("madNLP", mpc_para)
aux_sol     = aux_solver_economic("Gurobi", mpc_para)
admm_sol    = ADMM_eco_iter(mpc_para, prime_sol, aux_sol; tol = tol)

train_pool = [randn(mpc_para.nx) for i in 1:4]
test_pool  = [randn(mpc_para.nx) for i in 1:2]

# Training dataset
for x0 in train_pool
    println("================ Train x0[1:2] =",round.(x0[1:2], digits = 3),"================")
    println("================ $solver_name ================")
    u_opt, J_opt, solve_time = mpc_eco_sol(x0)
    @printf("J_opt = %5.3f, solving time = %5.2f ms\n", J_opt, solve_time*1000) 
    

    println("================ ADMM ================")
    u_ADMM, J_ADMM, _ =  admm_sol(; data = data_train, verbose = true)
    # @printf("J_ADMM = %5.3f\n", J_ADMM)
    @printf("ΔJ/J = %5.3f%%\n\n", abs(J_opt - J_ADMM) / abs(J_opt) * 100)
end
@printf("Collected %4d data points \n\n", length(data_train["input"]))
data = Dict("input" => reduce(hcat, data_train["input"]), "grad" => reduce(hcat, data_train["grad"]), "rho" => mpc_para.rho, "enve" => data_train["env"])
npzwrite(joinpath("data", string(typeof(cost_func),"-rho=", mpc_para.rho,"-train",".npz")), data)


# Testing dataset
data_test = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])
for x0 in test_pool
    println("================ Test x0[1:2] =",round.(x0[1:2], digits = 3),"================")
    u_ADMM, J_ADMM, _ =  admm_sol(; data = data_test, verbose = true)
end
@printf("Collected %4d data points \n\n", length(data_test["input"]))
data = Dict("input" => reduce(hcat, data_test["input"]), "grad" => reduce(hcat, data_test["grad"]), "rho" => mpc_para.rho, "enve" => data_test["env"])
npzwrite(joinpath("data", string(typeof(cost_func),"-rho=", mpc_para.rho,"-test",".npz")), data)