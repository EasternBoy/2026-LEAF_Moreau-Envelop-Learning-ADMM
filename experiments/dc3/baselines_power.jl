# Runs the repository's own economic-MPC solvers on the instances exported by
# the Python benchmark, and writes julia_baselines.json next to them.
#
#   julia --project=. experiments/dc3/baselines_power.jl DC3/results/power_grid
#
# See experiments/dc3/README.md for the Gurobi caveat.

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
const oracle_mode = length(ARGS) >= 3 && ARGS[3] == "oracle"
length(ARGS) < 3 || ARGS[3] in ("oracle", "deployment") || error("mode must be deployment or oracle")
const tol::FloatType = oracle_mode ? 1e-2 : 1e-6
include(joinpath(@__DIR__, "metrics.jl"))

GUROBI_ENV = nothing                   # skip Gurobi.Env() (no license here)
include(joinpath(REPO, "problems", "power_grid", "problem.jl"))
include(joinpath(REPO, "problems", "power_grid", "setup.jl"))
include(joinpath(REPO, "problems", "power_grid", "jump_solver.jl"))
include(joinpath(REPO, "problems", "power_grid", "lme_admm.jl"))

@assert Nw == mpc_data.N "instance horizon $(Nw) != energy_mag() N=$(mpc_data.N)"
@printf("power baselines: %d instances, N=%d, max_opt_gap=%.4g%%\n", count_, Nw, max_opt_gap)

J_hi = zeros(count_); t_hi = zeros(count_)
J_bm = zeros(count_); t_bm = zeros(count_)
J_sl = zeros(count_); t_sl = zeros(count_)
eq_sl = zeros(count_); in_sl = zeros(count_); dom_sl = zeros(count_)
metrics_hi = NamedTuple[]; metrics_bm = NamedTuple[]; metrics_sl = NamedTuple[]
iterations = zeros(Int, count_)

sol_hi = mpc_eco_solver("Ipopt", mpc_data, 1e-10)
mgrad   = gradient_struct(model, s_mb, dim; kernel = mmul_add_matrix!)
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

    vh, th, Jh = sol_hi(x0, load, gen; return_state=true)
    push!(metrics_hi, power_metrics(mpc_data, vh, x0, load, gen))
    J_hi[k] = Jh; t_hi[k] = th
    Jopt = Jh                                   # per-instance target for the callbacks

    sol_bm = mpc_eco_solver("Ipopt", mpc_data, tol)   # oracle mode alone enables the callback
    vb, tb, Jb = sol_bm(x0, load, gen; return_state=true)
    push!(metrics_bm, power_metrics(mpc_data, vb, x0, load, gen))
    J_bm[k] = Jb; t_bm[k] = tb

    # Track iterations; the deployment callback has no access to Jopt.
    cb = (z, w, alpha, v, beta, i, J) -> begin
        iterations[k] = i
        !oracle_mode || (isfinite(J) && 100abs(J - Jh) / abs(Jh) < max_opt_gap)
    end
    v, ts = admm(x0, load, gen, cb)
    met = power_metrics(mpc_data, v, x0, load, gen)
    push!(metrics_sl, met)
    J_sl[k] = met.obj; t_sl[k] = ts
    eq_sl[k] = met.eq; in_sl[k] = met.ineq
    pv = v[3, :]; dom_sl[k] = -minimum(pv)
    if k % 5 == 0; GC.gc(); end
    @printf("  [%3d/%3d] Jopt=%.4f  Ipopt_bm=%.4f (%.2f ms)  sLME=%.4f (%.2f ms, eq=%.1e, ineq=%.1e, min p=%.3g)\n",
            k, count_, Jh, Jb, 1e3tb, J_sl[k], 1e3ts, eq_sl[k], in_sl[k], minimum(pv))
end

result = Dict(
    "schema_version" => 2,
    "note" => "Measured original-constraint feasibility and exact objective domain. " *
              (oracle_mode ? "Oracle-assisted stopping; reference-solve cost excluded." : "Deployment stopping; no optimum oracle.") *
              " Gurobi-dependent baselines not run.",
    "n_instances" => count_, "max_opt_gap" => oracle_mode ? max_opt_gap : nothing,
    "methods" => Dict(
        "Ipopt(tol=1e-10)" => metric_summary(metrics_hi, t_hi, J_hi),
        (oracle_mode ? "Ipopt(oracle)" : "Ipopt(deployment)") => metric_summary(metrics_bm, t_bm, J_hi; oracle=oracle_mode),
        "LME-ADMM(split)" => merge(metric_summary(metrics_sl, t_sl, J_hi; oracle=oracle_mode),
            Dict("iterations_mean" => mean(iterations), "iterations_at_cap" => sum(iterations .>= 1000))),
        "ADMM(Gurobi aux)" => Dict("note" => "not_run: needs Gurobi")))

open(joinpath(out_dir, "julia_baselines.json"), "w") do f
    JSON3.pretty(f, result)
end
npzwrite(joinpath(out_dir, "julia_baselines.npz"),
         Dict("J_hi" => J_hi, "t_hi" => t_hi, "J_bm" => J_bm, "t_bm" => t_bm,
              "J_sl" => J_sl, "t_sl" => t_sl,
              "eq_sl" => eq_sl, "ineq_sl" => in_sl, "dom_sl" => dom_sl,
              "iterations" => iterations,
              "eq_hi" => [r.eq for r in metrics_hi], "ineq_hi" => [r.ineq for r in metrics_hi],
              "eq_bm" => [r.eq for r in metrics_bm], "ineq_bm" => [r.ineq for r in metrics_bm],
              "domain_valid_hi" => [r.valid for r in metrics_hi],
              "domain_valid_bm" => [r.valid for r in metrics_bm],
              "domain_valid_sl" => [r.valid for r in metrics_sl]))
println("wrote ", joinpath(out_dir, "julia_baselines.json"))
