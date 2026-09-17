using Pkg

Pkg.activate(".")
# Pkg.instantiate()

using LinearAlgebra, NPZ

include("cost_Func.jl")
include("MPC_JuMPsolver.jl")
include("system.jl")
include("MPC_ADMM.jl")

parameter = LTI_2D
const dict = parameter()
const cost_func = logsum
const max_iter = 100
const ρ = 1.

const tol = 1e-4

begin
    x0 = dict["x0"]
    
    println("\n================ MPC sparse: optimal input ================")
    mpc_spa = MPC_solver_gen("Ipopt")
    u_spa = optimize_para!(mpc_spa, x0, verbose= true)
    println(round.(u_spa, digits=4),"\n")

    mpc_spa = MPC_solver_gen("madNLP")
    u_spa = optimize_para!(mpc_spa, x0, verbose= true)
    println(round.(u_spa, digits=4))

    # println("\n================ MPC dense: optimal input ================")
    # mpc_den = MPCDense_solver_gen("madNLP")
    # u_den = optimize_fix!(mpc_den, x0, mode= "sumary")
    # println(round.(u_den, digits=4))

    println("\n================ ADMM: optimal input ================")
    prime_model = prime_solver("Ipopt")
    aux_model   = aux_solver("OSQP", x0)

    u_ADMM =  ADMM_iter(prime_model, aux_model; verbose = true)
    println(round.(u_ADMM, digits=4))
end
