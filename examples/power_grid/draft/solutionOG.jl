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
using Plots
using PrecompileTools


import ParametricOptInterface as POI
import MathOptInterface       as MOI

const FloatType = Float64
const tol_IPopt::FloatType = 1e-20
const tol_ADMM::FloatType  = 1e-10
const s_mb::Int = 24
const Ipopt_n_iter::Int = 40
const LADMM_n_iter::Int = 40
const sLADMM_n_iter::Int = 400





include("power_system.jl")
include("preprocess.jl")
include("eMPC_JuMPsolver.jl")
include("eMPC_ADMM.jl")
include("eMPC_L-ADMM.jl")



BESSinit = mpc_data.x0
load     = mpc_data.load_forecast[1:mpc_data.N]
gen      = mpc_data.gen_forecast[1:mpc_data.N]

begin
    cbs_Ipopt = callback_struct()
    mpc_eco_sol = mpc_eco_solver("Ipopt", mpc_data, tol_IPopt, cbs_Ipopt)
    _, sol_time_Ipopt = mpc_eco_sol(BESSinit, load, gen; verbose = true) 


    cbs_LME_ADMM = callback_struct()
    mgrad    = gradient_struct(model,  s_mb, dim)    
    aux_sol  = aux_solver_eco("Gurobi", mpc_data)
    admm_sol = LME_ADMM(mpc_data, mgrad, aux_sol)
    callback = (args...) -> ADMM_callback_iter(args..., cbs_LME_ADMM)
    _, sol_time_LME = admm_sol(BESSinit, load, gen, callback; verbose = true, tol = tol_ADMM)


    cbs_sLME_ADMM = callback_struct()
    mgrad         = gradient_struct(model,  s_mb, dim)    
    aux_sol       = dynamics_projection(mpc_data)
    admm_sol      = LME_ADMM_split(mpc_data, mgrad, aux_sol)
    callback = (args...) -> sADMM_callback_iter(args..., cbs_sLME_ADMM)
    _, sol_time_sLME = admm_sol(BESSinit, load, gen, callback; verbose = true, tol = tol_ADMM)
end


cbs_Ipopt = callback_struct()
mpc_eco_sol = mpc_eco_solver("Ipopt", mpc_data, tol_IPopt, cbs_Ipopt)
_, sol_time_Ipopt = mpc_eco_sol(BESSinit, load, gen) 


cbs_LME_ADMM = callback_struct()
mgrad    = gradient_struct(model,  s_mb, dim)    
aux_sol  = aux_solver_eco("Gurobi", mpc_data)
admm_sol = LME_ADMM(mpc_data, mgrad, aux_sol)
callback = (args...) -> ADMM_callback_iter(args..., cbs_LME_ADMM)
_, sol_time_LME = admm_sol(BESSinit, load, gen, callback; tol = tol_ADMM)


cbs_sLME_ADMM = callback_struct()
mgrad         = gradient_struct(model,  s_mb, dim)    
aux_sol       = dynamics_projection(mpc_data)
admm_sol      = LME_ADMM_split(mpc_data, mgrad, aux_sol)
callback = (args...) -> sADMM_callback_iter(args..., cbs_sLME_ADMM)
_, sol_time_sLME = admm_sol(BESSinit, load, gen, callback; tol = tol_ADMM)

cbs_LME_ADMM.n_iter  = [0; cbs_LME_ADMM.n_iter];  cbs_LME_ADMM.rel_opt_gap  = [cbs_Ipopt.rel_opt_gap[1]; cbs_LME_ADMM.rel_opt_gap]
cbs_sLME_ADMM.n_iter = [0; cbs_sLME_ADMM.n_iter]; cbs_sLME_ADMM.rel_opt_gap = [cbs_Ipopt.rel_opt_gap[1]; cbs_sLME_ADMM.rel_opt_gap]

p = plot(size = (600, 350),
        ytick = [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
        yscale = :log10, xlims = [0, 20], 
        xlabel = "Time (ms)", ylabel = "Optimality gap (%)", 
        titlefont = 18,         # title font size
        guidefont = 14,         # axis labels (xlabel/ylabel)
        tickfont = 10,          # tick labels
        legendfontsize = 10)


plot!(1000*cbs_Ipopt.n_iter*sol_time_Ipopt/Ipopt_n_iter,    cbs_Ipopt.rel_opt_gap, label = "IPopt", linewidth = 2)
plot!(1000*cbs_LME_ADMM.n_iter*sol_time_LME/LADMM_n_iter,   cbs_LME_ADMM.rel_opt_gap, label = "LME-ADMM", linewidth = 2)
plot!(1000*cbs_sLME_ADMM.n_iter*sol_time_sLME/sLADMM_n_iter, cbs_sLME_ADMM.rel_opt_gap, label = "sLME-ADMM", linewidth = 2)

savefig(p, "media/figures/OptGap_time.pdf")


