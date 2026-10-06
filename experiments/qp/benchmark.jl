using Pkg

Pkg.activate(".")
Pkg.instantiate()

using Printf
using SparseArrays
using LinearAlgebra
using NPZ
using Distributions, Statistics
using Random
using JuMP
using OSQP
using Plots
using StatsPlots

const FloatType = Float64

const n::Int = 100
const neq::Int = 50
const m::Int = 50
const tol::FloatType = 1e-3
const max_iter::Int = 1000
const max_opt_gap::FloatType = 1.0

nsamples = 833
cl_gc    = 5
Random.seed!(20260923)

include("../../problems/qp/problem.jl")
include("../../problems/qp/setup.jl")
include("../../problems/qp/jump_solver.jl")
include("../../problems/qp/lme_admm.jl")

warm_data = data_opt(n, neq, m)
@assert isdiag(warm_data.Q) && all(diag(warm_data.Q) .>= 0)
@assert size(warm_data.Q) == (n, n) && length(warm_data.h) == m
@assert rank(warm_data.A) == neq
@assert rho ≈ warm_data.rho

mgrad = gradient_struct(model, 1, n)
stime_OSQP      = FloatType[]
stime_sLME_ADMM = FloatType[]
J_OSQP         = FloatType[]
J_sLME_ADMM    = FloatType[]
opt_gap        = FloatType[]
eq_viol        = FloatType[]
ineq_viol      = FloatType[]
mean_eq_viol   = FloatType[]
mean_ineq_viol = FloatType[]

# Warm up both methods before recording timings.
_, _, J_opt = JuMP_solver("osqp", warm_data, 1e-8)
sLME_ADMM(warm_data, mgrad, sLME_ADMM_callback; tol = tol, max_iter = max_iter)

for k in 1:nsamples
    global J_opt
    para_opt = data_opt(n, neq, m)
    BL_sol_opt, solve_time_OSQP, J_opt = JuMP_solver("osqp", para_opt, 1e-8)
    push!(stime_OSQP, solve_time_OSQP)
    push!(J_OSQP, J_opt)
    println("Ground truth optimal objective value (baseline): $J_opt")
    if k % cl_gc == 0 GC.gc() end

    sol, solve_time_sLME_ADMM, J = sLME_ADMM(para_opt, mgrad, sLME_ADMM_callback; tol = tol, max_iter = max_iter)
    push!(stime_sLME_ADMM, solve_time_sLME_ADMM)
    push!(J_sLME_ADMM, J)
    push!(opt_gap, 100abs(J - J_opt)/max(abs(J_opt), eps(FloatType)))
    eq_residual = abs.(para_opt.A*sol - para_opt.x)
    ineq_residual = max.(para_opt.G*sol - para_opt.h, 0.0)
    push!(eq_viol, maximum(eq_residual))
    push!(ineq_viol, maximum(ineq_residual))
    push!(mean_eq_viol, mean(eq_residual))
    push!(mean_ineq_viol, mean(ineq_residual))
    println("opt_gap between sLME_ADMM optimal objective and baseline: $(round(opt_gap[end], digits = 3))%")
    println("Equality residual: $(eq_viol[end]), inequality violation: $(ineq_viol[end])")
    println()
    if k % cl_gc == 0 GC.gc() end
end

data = Dict("OSQP"       => (stime_OSQP      *= 1e3),  # second -> millisecond
            "sLME_ADMM"  => (stime_sLME_ADMM *= 1e3),
            "J_OSQP"     => J_OSQP,
            "J_sLME_ADMM" => J_sLME_ADMM,
            "opt_gap"    => opt_gap,
            "eq_viol"    => eq_viol,
            "ineq_viol"  => ineq_viol,
            "mean_eq_viol" => mean_eq_viol,
            "mean_ineq_viol" => mean_ineq_viol,
            "tol" => tol, "max_opt_gap" => max_opt_gap,
            "max_iter" => max_iter, "oracle_assisted" => true)

bp = boxplot(stime_OSQP, size=(400, 600),
             framestyle = :box,
             outliers = false,
             legend = false,
             xticks = (1:2, ["OSQP", "sLME-ADMM"]),
             tickfont = 16, guidefont = 16, legendfont = 16)
boxplot!(stime_sLME_ADMM, legend = false, outliers = false)

if nsamples >= 833
    mkpath(joinpath("results", "qp", "benchmark"))
    mkpath(joinpath("results", "qp", "figures"))
    npzwrite(joinpath("results", "qp", "benchmark", string("qp-n=", n, "neq=", neq, "m=", m, "-tol=", tol, "-gopt=", max_opt_gap, ".npz")), data)
    savefig(bp, joinpath("results", "qp", "figures", string("qp-n=", n, "neq=", neq, "m=", m, "-tol=", tol, "-gopt=", max_opt_gap, ".pdf")))
end

println("OSQP mean (ms): ", mean(stime_OSQP))
println("OSQP mean objective: ", mean(J_OSQP))
println("sLME-ADMM objective gap (%): mean=", mean(opt_gap), ", max=", maximum(opt_gap))
println("sLME-ADMM runs with objective gap > $max_opt_gap%: ", count(opt_gap .> max_opt_gap))
@printf("%-16s %12s %12s %12s %12s %12s %12s\n",
        "Method", "Obj. value", "Max eq.", "Mean eq.", "Max ineq.", "Mean ineq.", "Time (s)")
@printf("%-16s %12.5f %12.5e %12.5e %12.5e %12.5e %12.6f\n",
        "sLME-ADMM", mean(J_sLME_ADMM), mean(eq_viol), mean(mean_eq_viol),
        mean(ineq_viol), mean(mean_ineq_viol), mean(stime_sLME_ADMM)/1e3)
