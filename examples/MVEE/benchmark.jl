using Pkg

Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra, SparseArrays
using Printf
using Distributed
using JSON3
using NNlib
using Distributions, Statistics
using NPZ
using Base.Threads
using PyPlot
using StaticArrays
using MathOptInterface
using ParametricOptInterface
using IterTools
using CSV, DataFrames
using Convex
using Random
using PyPlot
using LDLFactorizations
using StatsPlots
using Statistics
using ProgressMeter

using JuMP
using OSQP
using Gurobi
using ECOS
using SCS
using Clarabel
using MosekTools, Mosek

const MOI = MathOptInterface
const POI = ParametricOptInterface
const FloatType = Float64
const max_opt_gap::FloatType = 0.1

pygui(true)
PyPlot.using3D()

include("logdet_utils.jl")
include("logdet_system.jl")                                           
include("logdet_JuMPSolver.jl")
include("logdet_L-ADMM.jl")
include("logdet_preprocess.jl") 

# ================================ Load data ==============================

rho, mp = load_model("data/MVEE_data/logdet-rho=1.0-m=50_max_ICNN.json")
# rho, mp = load_model("examples/MVEE/logdet_mpc-rho=1.0-m=50_ICNN.json")

model = ICNN(
    mp.U[1], 
    mp.b[1],
    [ICNN_Layer(mp.U[i], mp.W[i], mp.b[i]) for i in 2:length(mp.U)],
    mp.v, 
    mp.a,
    mp.c)

# # ================================ Benchmarking ==============================
data_int         = data_opt()
solver_Clarabel  = logdet_solver("Clarabel", data_int)
solver_Mosek     = logdet_solver("Mosek", data_int)
mgrad            = gradient_struct(model, 1, div(data_int.n*(data_int.n+1),2))

N_samples = 10000 # Match the loop count

solve_time_Mosek    = FloatType[]
solve_time_Clarabel = FloatType[]
solve_time_LME_ADMM = FloatType[]

println("Collecting $N_samples samples...")

p = Progress(N_samples, 1, "Collecting data: ")

for i in 1:N_samples
    # println("---------------- Clarabel ----------------")
    data  = data_opt()
    sol_Clarabel, time_Clarabel, obj_Clarabel = solver_Clarabel(data)
    global Jopt = obj_Clarabel
    push!(solve_time_Clarabel, time_Clarabel)
    # println("Objective value = ", exp(obj_Clarabel))
    # println("Solve time = ",      time_Clarabel)
    # println("")
    # println("---------------- Mosek ----------------")
    sol_GT, time_Mosek, obj_GT = solver_Mosek(data)
    push!(solve_time_Mosek, time_Mosek)
    # println("Objective value = ", exp(obj_GT))
    # println("Solve time = ",      time_Mosek)
    # println("")
    # println("---------------- sLME-ADMM ----------------")
    sol_LME, time_LME, obj_LME = sLME_ADMM(data, mgrad, sLME_ADMM_callback)
    push!(solve_time_LME_ADMM, time_LME)
    # println("Objective value = ", exp(obj_LME))
    gap = abs(exp(obj_Clarabel) - exp(obj_LME))/(exp(obj_Clarabel))*100
    if gap > 5.0
        println("Warning: Optimal gap too large: ", gap)
        continue
    end
    # println("Optimal gap of sLME_ADMM = ", abs(exp(obj_Clarabel) - exp(obj_LME))/(exp(obj_Clarabel))*100)
    # println("Solve time of sLME_ADMM = ",  time_LME)
    # println("")
    # plot_ellipsoid(sol_LME, sol_GT, data.A, exp(obj_GT), exp(obj_LME); x_scale=1., y_scale=1.)
    GC.gc()
    next!(p)
                                                                                                                                                                                
end


df_arr = DataFrame(
    Mosek    = solve_time_Mosek[2:end]*1e3,
    Clarabel = solve_time_Clarabel[2:end]*1e3,
    LME_ADMM = solve_time_LME_ADMM[2:end]*1e3)

output_folder = joinpath("data", "MVEE_data", "benchmark_results")


filename_arr = joinpath(output_folder, "benchmark_results_N=$(N_samples)_m=$(data_int.m)-rho=$(rho)_optgap=$(max_opt_gap).csv")
CSV.write(filename_arr, df_arr)

# bp = boxplot(solve_time_Mosek, size=(550, 600),
#     labels = "Mosek",
#     framestyle = :box,
#     outliers = false,
#     xticks = (1:3, ["", "", ""]),
#     tickfont = 16,
#     guidefont = 16,
#     legendfont = 16)
# boxplot!(solve_time_Clarabel, label = "Clarabel", outliers=false)
# boxplot!(solve_time_LME_ADMM, label = "sLME-ADMM", outliers=false)

# if N_samples >= 1000
#     npzwrite(joinpath("data", "MVEE_data", "benchmark_results", string("logdet_m=", m,"-opt_gap=",max_opt_gap,".npz")), data)
#     savefig(bp, joinpath("media","figures",   string("logdet_m=", m, "opt_gap=",max_opt_gap,".pdf")))
# end


# println("Mosek: ",     median(solve_time_Mosek[2:end]))
# println("Clarabel: ",  median(solve_time_Clarabel[2:end]))
# println("sLME-ADMM: ", median(solve_time_LME_ADMM[2:end]))