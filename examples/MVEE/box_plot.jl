using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CSV
using DataFrames
using Plots
using StatsPlots
using Printf
using Statistics

data_arr = CSV.read("data/MVEE_data/benchmark_results/benchmark_results_N=10000_m=55-rho=1.0_optgap=1.0.csv", DataFrame)

# ================================== Plotting ==================================
bp = boxplot(data_arr.Mosek, size=(550, 600),
    labels = "Mosek",
    framestyle = :box,
    outliers = false,
    xticks = (1:3, ["", "", ""]),
    tickfont = 16,
    guidefont = 16,
    legendfont = 16,
    legend = :topright)
boxplot!(data_arr.Clarabel, label = "Clarabel", outliers=false, legend = :topright)
boxplot!(data_arr.LME_ADMM, label = "sLME-ADMM", outliers=false, legend = :topright)

savefig(bp, joinpath("media","figures",   string("logdet_m=50",".pdf")))

println("Mosek: ",     median(data_arr.Mosek))
println("Clarabel: ",  median(data_arr.Clarabel))
println("sLME-ADMM: ", median(data_arr.LME_ADMM))