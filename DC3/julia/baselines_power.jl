# Runs the repository's own economic-MPC solvers on the instances exported by
# the Python benchmark, and writes julia_baselines.json next to them.
#
#   julia --project=. DC3/julia/baselines_power.jl DC3/results/power_grid
#
# See DC3/julia/README.md for the Gurobi caveat.

using Pkg; Pkg.activate(".")
using Base.Threads
using Printf, SparseArrays, JSON3, LinearAlgebra, StaticArrays, NPZ, JuMP, NNlib
using Distributions, Statistics
import ParametricOptInterface as POI
import MathOptInterface       as MOI

const FloatType = Float64
const REPO = abspath(joinpath(@__DIR__, "..", ".."))

out_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "DC3", "results", "power_grid")
out_dir = isabspath(out_dir) ? out_dir : joinpath(REPO, out_dir)
inst    = npzread(joinpath(out_dir, "test_instances.npz"))

x0s   = inst["x0"]; loads = inst["load"]; gens = inst["gen"]
count_ = length(x0s); Nw = size(loads, 2)

const max_opt_gap::FloatType = length(ARGS) >= 2 ? parse(FloatType, ARGS[2]) : 0.01
const s_mb::Int = 24
const tol::FloatType = 1e-2

GUROBI_ENV = nothing                   # skip Gurobi.Env() (no license here)
include(joinpath(REPO, "examples", "power_grid", "power_system.jl"))
include(joinpath(REPO, "examples", "power_grid", "preprocess.jl"))
include(joinpath(REPO, "examples", "power_grid", "eMPC_JuMPsolver.jl"))
include(joinpath(REPO, "examples", "power_grid", "eMPC_L-ADMM.jl"))

@assert Nw == mpc_data.N "instance horizon $(Nw) != energy_mag() N=$(mpc_data.N)"
@printf("power baselines: %d instances, N=%d, max_opt_gap=%.4g%%\n", count_, Nw, max_opt_gap)

J_hi = zeros(count_); t_hi = zeros(count_)
J_bm = zeros(count_); t_bm = zeros(count_)
J_sl = zeros(count_); t_sl = zeros(count_)
eq_sl = zeros(count_); in_sl = zeros(count_); dom_sl = zeros(count_)

sol_hi = mpc_eco_solver("Ipopt", mpc_data, 1e-10)
mgrad   = gradient_struct(model, s_mb, dim)
aux_sol = dynamics_projection(mpc_data)
admm    = LME_ADMM_split(mpc_data, mgrad, aux_sol)

A_ = mpc_data.A; B_ = mpc_data.B
x_min = mpc_data.x_min; x_max = mpc_data.x_max
u_min = mpc_data.u_min; u_max = mpc_data.u_max

for k in 1:count_
    global Jopt
    x0   = FloatType(x0s[k])
    load = Vector{FloatType}(loads[k, :])
    gen  = Vector{FloatType}(gens[k, :])

    _, th, Jh = sol_hi(x0, load, gen)
    J_hi[k] = Jh; t_hi[k] = th
    Jopt = Jh                                   # per-instance target for the callbacks

    sol_bm = mpc_eco_solver("Ipopt", mpc_data, tol)   # tol>=1e-3 -> Ipopt_callback_BM
    _, tb, Jb = sol_bm(x0, load, gen)
    J_bm[k] = Jb; t_bm[k] = tb

    # default tol / max_iter, exactly as examples/power_grid/benchmarkOG.jl calls it
    v, ts = admm(x0, load, gen, sLME_ADMM_callback)
    mv = v[1, :]; uv = v[2, :]; pv = v[3, :]; xv = v[4, :]
    J_sl[k] = sum(mpc_data.cost_func(mv[i], uv[i], max(pv[i], 1e-12)) for i in 1:Nw)
    t_sl[k] = ts
    # residuals of the *original* constraints (x reconstructed from u, as in the model)
    xr = zeros(Nw + 1); xr[1] = x0
    for i in 1:Nw; xr[i+1] = A_*xr[i] + B_*uv[i]; end
    eq_sl[k] = max(maximum(abs.(uv .+ mv .+ gen .- load .- pv)),
                   abs(xr[end] - x0), maximum(abs.(xr[2:end] .- xv)))
    in_sl[k] = maximum([maximum(uv .- u_max), maximum(u_min .- uv), maximum(-pv),
                        maximum(xr .- x_max), maximum(x_min .- xr), 0.0])
    dom_sl[k] = -minimum(pv)
    if k % 5 == 0; GC.gc(); end
    @printf("  [%3d/%3d] Jopt=%.4f  Ipopt_bm=%.4f (%.2f ms)  sLME=%.4f (%.2f ms, eq=%.1e, ineq=%.1e, min p=%.3g)\n",
            k, count_, Jh, Jb, 1e3tb, J_sl[k], 1e3ts, eq_sl[k], in_sl[k], minimum(pv))
end

gap(J, Jr) = 100 .* abs.(J .- Jr) ./ abs.(Jr)
feas(e, i, d) = mean((e .<= 1e-4) .& (i .<= 1e-4) .& (d .<= 1e-4))

result = Dict(
  "note" => "produced by DC3/julia/baselines_power.jl on DC3's exported test instances; " *
            "Gurobi-based baselines (standard ADMM and LME_ADMM with aux_solver_eco) " *
            "were not run - no license.",
  "n_instances" => count_,
  "max_opt_gap" => max_opt_gap,
  "methods" => Dict(
    "Ipopt(tol=1e-10)" => Dict(
        "obj_mean" => mean(J_hi), "gap_pct_mean" => 0.0, "gap_pct_max" => 0.0,
        "latency_median_ms" => 1e3median(t_hi), "feasible_rate" => 1.0,
        "note" => "ground-truth reference (examples/power_grid/solutionOG.jl uses tol 1e-20)"),
    "Ipopt(early-stop)" => Dict(
        "obj_mean" => mean(J_bm),
        "gap_pct_mean" => mean(gap(J_bm, J_hi)), "gap_pct_max" => maximum(gap(J_bm, J_hi)),
        "latency_median_ms" => 1e3median(t_bm), "feasible_rate" => 1.0,
        "note" => "Ipopt stopped by Ipopt_callback_BM at $(max_opt_gap)% relative gap"),
    "LME-ADMM(split)" => Dict(
        "obj_mean" => mean(J_sl),
        "gap_pct_mean" => mean(gap(J_sl, J_hi)), "gap_pct_max" => maximum(gap(J_sl, J_hi)),
        "latency_median_ms" => 1e3median(t_sl),
        "eq_max" => maximum(eq_sl), "ineq_max" => maximum(in_sl),
        "feasible_rate" => feas(eq_sl, in_sl, dom_sl),
        "note" => "examples/power_grid/eMPC_L-ADMM.jl :: LME_ADMM_split with dynamics_projection"),
    "ADMM(Gurobi aux)" => Dict("note" => "not_run: needs Gurobi for aux_solver_eco"),
  ))

open(joinpath(out_dir, "julia_baselines.json"), "w") do f
    JSON3.pretty(f, result)
end
npzwrite(joinpath(out_dir, "julia_baselines.npz"),
         Dict("J_hi" => J_hi, "t_hi" => t_hi, "J_bm" => J_bm, "t_bm" => t_bm,
              "J_sl" => J_sl, "t_sl" => t_sl,
              "eq_sl" => eq_sl, "ineq_sl" => in_sl, "dom_sl" => dom_sl))
println("wrote ", joinpath(out_dir, "julia_baselines.json"))
