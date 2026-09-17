using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Printf
using LinearAlgebra
using SparseArrays
using NPZ
using JSON3
using NNlib
using Distributions, Statistics
using Base.Threads
using JuMP
using Clarabel
using Plots
using StatsPlots
# export JULIA_NUM_THREADS=8
import ParametricOptInterface as POI
import MathOptInterface       as MOI

include("mpc_system.jl")
include("utils.jl")
include("mpc_JuMPsolver.jl")
include("mpc_ADMM.jl")
include("mpc_LME-ADMM.jl")

# ── Problem constants ──
para_opt = MPCData()
const n   = (para_opt.T + 1) * (para_opt.nx + para_opt.nu)  
const s_mb::Int = para_opt.nx + para_opt.nu   

const max_opt_gap::FloatType = 0.05

nsamples = 5
cl_gc    = 5

# ── Build solvers ──
admm_sol = ADMM_mpc(para_opt, "Clarabel"; tol=1e-3, max_iter=500)
aux_sol  = aux_solver_mpc_lme(para_opt)
mgrad    = gradient_struct(model, 1, s_mb) 
lme_sol  = LME_ADMM_mpc(para_opt, mgrad, aux_sol)

stime_Clarabel    = FloatType[]
stime_ADMM      = FloatType[]
stime_LME_ADMM  = FloatType[]
stime_sLME_ADMM = FloatType[]

for k in 1:nsamples

    global J_opt

    new_para = new_instance(para_opt)

    # ── Baseline ──
    _, _, J_opt = JuMP_solver("Clarabel", new_para, 1e-8)
    println("Baseline J_opt = $J_opt")

    # ── Clarabel ──
    _, solve_time_Clarabel, J_Clarabel = JuMP_solver("Clarabel", new_para, 1e-4)
    push!(stime_Clarabel, solve_time_Clarabel)
    @printf("Clarabel    ΔJ/J = %5.3f%%  t = %5.2f ms\n",
            100abs((J_opt - J_Clarabel)/J_opt), solve_time_Clarabel*1e3)

    # ── Conventional ADMM ──
    t_start = time_ns()
    _, J_ADMM = admm_sol(nothing, new_para)
    solve_time_ADMM = (time_ns() - t_start) / 1e9
    push!(stime_ADMM, solve_time_ADMM)
    @printf("ADMM      ΔJ/J = %5.3f%%  t = %5.2f ms\n",
            100abs((J_opt - J_ADMM)/J_opt), solve_time_ADMM*1e3)

    # ── LME-ADMM ──
    _, solve_time_LME, J_LME = lme_sol(new_para; tol=1e-3, max_iter=500)
    push!(stime_LME_ADMM, solve_time_LME)
    @printf("LME-ADMM  ΔJ/J = %5.3f%%  t = %5.2f ms\n",
            100abs((J_opt - J_LME)/J_opt), solve_time_LME*1e3)

    # ── sLME-ADMM ──
    _, solve_time_sLME, J_sLME = sLME_ADMM_mpc(new_para, mgrad; tol=1e-3, max_iter=500)
    push!(stime_sLME_ADMM, solve_time_sLME)
    @printf("sLME-ADMM ΔJ/J = %5.3f%%  t = %5.2f ms\n",
            100abs((J_opt - J_sLME)/J_opt), solve_time_sLME*1e3)

    println()
    if k % cl_gc == 0  GC.gc()  end
end

# ── Convert to ms ──
stime_Clarabel    .*= 1e3
stime_ADMM      .*= 1e3
stime_LME_ADMM  .*= 1e3
stime_sLME_ADMM .*= 1e3

println("\n===== Median solve time (ms) =====")
@printf("Clarabel:    %6.2f ms\n", median(stime_Clarabel[2:end]))
@printf("ADMM:      %6.2f ms\n", median(stime_ADMM[2:end]))
@printf("LME-ADMM:  %6.2f ms\n", median(stime_LME_ADMM[2:end]))
@printf("sLME-ADMM: %6.2f ms\n", median(stime_sLME_ADMM[2:end]))


bp = boxplot(stime_Clarabel,
             label="Clarabel", size=(500, 600),
             framestyle=:box, outliers=false,
             xticks=(1:4, ["Clarabel", "ADMM", "LME-ADMM", "sLME-ADMM"]),
             tickfont=14, guidefont=14, legendfont=14,
             ylabel="Solve time (ms)")
boxplot!(stime_ADMM,      label="ADMM",      outliers=false)
boxplot!(stime_LME_ADMM,  label="LME-ADMM",  outliers=false)
boxplot!(stime_sLME_ADMM, label="sLME-ADMM", outliers=false, legend=:best)

if nsamples >= 100
    data = Dict(
        "Clarabel"    => stime_Clarabel,
        "ADMM"      => stime_ADMM,
        "LME_ADMM"  => stime_LME_ADMM,
        "sLME_ADMM" => stime_sLME_ADMM,
    )
    tag = string("mpc-nx=", para_opt.nx, "-T=", para_opt.T, "-rho=", para_opt.rho)
    npzwrite(joinpath("data", "solving_data", tag * ".npz"), data)
    savefig(bp, joinpath("media", "figures", tag * ".pdf"))
end
