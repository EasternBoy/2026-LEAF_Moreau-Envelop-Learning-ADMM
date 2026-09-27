using Pkg

Pkg.activate(".")
Pkg.instantiate()

using AppleAccelerate
using Printf
using BenchmarkTools
using Profile
using BlockArrays
using SparseArrays
using NNlib
using JSON3
using LinearAlgebra
using StaticArrays
using NPZ
using Distributions, Statistics
using Base.Threads
using Printf
using JuMP


import ParametricOptInterface as POI
import MathOptInterface       as MOI

const FloatType = Float64
const tol::FloatType = 1e-2
const s_mb::Int = 24
const max_opt_gap::FloatType = 0.01


include("../../problems/power_grid/problem.jl")
include("../../problems/power_grid/setup.jl")
include("../../problems/power_grid/jump_solver.jl")
include("../../problems/power_grid/admm.jl")
include("../../problems/power_grid/lme_admm.jl")



BESSinit = mpc_data.x0
load     = mpc_data.load_forecast[1:mpc_data.N]
gen      = mpc_data.gen_forecast[1:mpc_data.N]


# samples = 300
# println("=============== Benchmarking solver Ipopt: \033[1m N = $N, relative optimality gap = $max_opt_gap% \033[0m ===============")
# mpc_eco_sol = mpc_eco_solver("Ipopt", mpc_data, tol)
# mpc_eco_sol(BESSinit, load, gen; verbose = true)
# stats = get_benchmark(mpc_eco_sol, (BESSinit, load, gen), samples)
# @printf("Time  (mean ± σ):  %5.3f ms ± %5.3f ms\n", stats.μ*1000, stats.σ*1000)



# samples = 100
# println("=============== Benchmarking solver madNLP: \033[1m N = $N, relative optimality gap = $max_opt_gap% \033[0m ===============")
# mpc_eco_sol = mpc_eco_solver("MadNLP", mpc_data, tol)
# mpc_eco_sol(BESSinit, load, gen; verbose = true)
# stats = get_benchmark(mpc_eco_sol, (BESSinit, load, gen), samples)
# @printf("Time  (mean ± σ):  %5.3f ms ± %5.3f ms\n", stats.μ*1000, stats.σ*1000)


samples   = 100
println("=============== Benchmarking ADMM with Ipopt (prime) and Gurobi (auxiliary): \033[1m #samples = $samples, N = $N, relative optimality gap = $max_opt_gap% \033[0m ===============")
prime_sol  = prime_sol_struct("Ipopt", mpc_data)
aux_sol    = aux_solver_eco("Gurobi", mpc_data)
admm_sol   = ADMM_eco_iter(mpc_data, prime_sol, aux_sol)
admm_sol(BESSinit, load, gen, ADMM_callback; verbose = true)
stats = get_benchmark(admm_sol, (BESSinit, load, gen, ADMM_callback), samples)
@printf("Time  (mean ± σ):  %5.3f ms ± %5.3f ms\n", stats.μ*1000, stats.σ*1000)


samples = 300
println("========== Benchmarking LME-ADMM: \033[1m #samples = $samples, N = $N, relative optimality gap = $max_opt_gap% \033[0m ===========")
mgrad    = gradient_struct(model, s_mb, dim; kernel = mmul_add_matrix!)    
aux_sol  = aux_solver_eco("Gurobi", mpc_data)
admm_sol = LME_ADMM(mpc_data, mgrad, aux_sol)
admm_sol(BESSinit, load, gen, ADMM_callback; verbose = true)
stats = get_benchmark(admm_sol, (BESSinit, load, gen, ADMM_callback), samples)
@printf("Time  (mean ± σ):  %5.3f ms ± %5.3f ms\n", stats.μ*1000, stats.σ*1000)



samples = 300
println("========== Benchmarking LME-ADMM with spliting: \033[1m number of samples = $samples, N = $N, relative optimality gap = $max_opt_gap% \033[0m ============")
mgrad    = gradient_struct(model, s_mb, dim; kernel = mmul_add_matrix!)
aux_sol  = dynamics_projection(mpc_data)
admm_sol = LME_ADMM_split(mpc_data, mgrad, aux_sol)
admm_sol(BESSinit, load, gen, sLME_ADMM_callback; verbose = true)
display(@benchmark admm_sol($BESSinit, $load, $gen, $sLME_ADMM_callback) samples=samples)