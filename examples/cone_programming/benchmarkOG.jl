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
using OSQP
using Plots
using StatsPlots

import ParametricOptInterface as POI
import MathOptInterface       as MOI

const FloatType = Float64

const n::Int = 1000
const m::Int = 100

const max_opt_gap::FloatType = 0.01

const s_mb::Int = max(div(n, Threads.nthreads())+1, 50)
nsamples = 1000
cl_gc    = 5


include("maxEntropy.jl")
include("preprocess.jl")
include("JuMPsolver.jl")
include("mADMM.jl")
include("LME-ADMM.jl")

mgrad = gradient_struct(model,  s_mb, 1)    
stime_IPOPT      = FloatType[]
stime_sLME_ADMM  = FloatType[]

for k in 1:nsamples

    global J_opt

    para_opt = data_opt(n, m)
    BL_sol_opt, _, J_opt = JuMP_solver("Ipopt", para_opt, 1e-8)
    println("Ground truth optimal objective value (baseline): $J_opt")
    if k % cl_gc == 0 GC.gc() end

    _, solve_time_IPopt, J_ipopt = JuMP_solver("Ipopt", para_opt, 1e-2, callback_struct())
    push!(stime_IPOPT, solve_time_IPopt)
    println("opt_gap between IPOPT optimal objective and baseline: $(round(100abs((J_opt - J_ipopt)/J_opt), digits = 3))%")
    if k % cl_gc == 0 GC.gc() end


    sol, solve_time_sLME_ADMM, J_sLME_ADMM = sLME_ADMM(para_opt, mgrad, sLME_ADMM_callback) #Remove verbose = true to disable print
    println("opt_gap between sLME_ADMM optimal objective and baseline: $(round(100abs((J_opt - J_sLME_ADMM)/J_opt), digits = 3))%")

    push!(stime_sLME_ADMM, solve_time_sLME_ADMM)
    println()
    if k % cl_gc == 0 GC.gc() end
end

data = Dict("IPOPT"     => (stime_IPOPT     *= 1e3),  #second -> milisecond
            "sLME_ADMM" => (stime_sLME_ADMM *= 1e3))


bp = boxplot(stime_IPOPT, size=(400, 600),
             framestyle = :box,
             outliers = false, 
             legend = false,
             xticks = (1:5, ["IPOPT", "sLME-ADMM",]),
             tickfont = 16, guidefont = 16, legendfont = 16
             )
boxplot!(stime_sLME_ADMM, legend = false, outliers=false)

if nsamples >= 1000
    npzwrite(joinpath("data", "solving_data", string("maxEntropy-n=", n, "m=", m,"-opt_gap=",max_opt_gap,".npz")), data)
    savefig(bp, joinpath("media","figures",   string("maxEntropy-n=", n, "m=", m, "opt_gap=",max_opt_gap,".pdf")))
end


println("IPOPT: ",     median(stime_IPOPT[2:end]))
println("sLME-ADMM: ", median(stime_sLME_ADMM[2:end]))