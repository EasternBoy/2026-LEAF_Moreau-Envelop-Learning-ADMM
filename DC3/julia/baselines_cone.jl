# Runs the repository's own cone-programming solvers on the instances exported
# by the Python benchmark, and writes julia_baselines.json next to them.
#
#   julia --project=. DC3/julia/baselines_cone.jl DC3/results/entr_max
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

out_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "DC3", "results", "entr_max")
out_dir = isabspath(out_dir) ? out_dir : joinpath(REPO, out_dir)
inst    = npzread(joinpath(out_dir, "test_instances.npz"))

Ainst = inst["A"]                      # (count, m, n)
binst = inst["b"]                      # (count, m)
count_, m_, n_ = size(Ainst)

const oracle_mode = length(ARGS) >= 3 && ARGS[3] == "oracle"
length(ARGS) < 3 || ARGS[3] in ("oracle", "deployment") || error("mode must be deployment or oracle")
include(joinpath(@__DIR__, "metrics.jl"))
const n::Int = n_
const m::Int = m_
const max_opt_gap::FloatType = length(ARGS) >= 2 ? parse(FloatType, ARGS[2]) : 0.1
# experiments/entr_max/benchmark.jl uses `max(div(n, nthreads())+1, 50)`.  That formula exceeds `n`
# when Julia runs single-threaded (or when n is small), and problems/entr_max/
# utils.jl::mini_batch then indexes out of bounds - so it is additionally clamped to n.
# Start Julia with `--threads=auto` to reproduce the multi-threaded setting.
const s_mb::Int = min(max(div(n, max(Threads.nthreads(), 1)) + 1, 50), n)

GUROBI_ENV = nothing                   # skip Gurobi.Env() (no license here)
include(joinpath(REPO, "problems", "entr_max", "problem.jl"))
include(joinpath(REPO, "problems", "entr_max", "setup.jl"))
include(joinpath(REPO, "problems", "entr_max", "jump_solver.jl"))
include(joinpath(REPO, "problems", "entr_max", "lme_admm.jl"))

@printf("cone baselines: %d instances, n=%d, m=%d, max_opt_gap=%.4g%%\n", count_, n, m, max_opt_gap)

mgrad = gradient_struct(model, s_mb, 1)

J_ipopt_hi = zeros(count_);  t_ipopt_hi = zeros(count_)
J_ipopt_bm = zeros(count_);  t_ipopt_bm = zeros(count_);  viol_bm = zeros(count_)
J_slme     = zeros(count_);  t_slme     = zeros(count_)
eq_slme    = zeros(count_);  in_slme    = zeros(count_)
W_slme     = zeros(count_, n)
it_slme    = zeros(Int, count_)
metrics_hi = NamedTuple[]; metrics_bm = NamedTuple[]; metrics_sl = NamedTuple[]

for k in 1:count_
    global J_opt
    A = Matrix{FloatType}(Ainst[k, :, :])
    b = Vector{FloatType}(binst[k, :])
    para = data_opt(n, m, A, b, 1.0, x -> x * log(x))

    # ground truth (tol 1e-8), exactly as in experiments/entr_max/benchmark.jl
    wh, t_hi, J_opt = JuMP_solver("Ipopt", para, 1e-8)
    push!(metrics_hi, cone_metrics(A, b, wh))
    J_ipopt_hi[k] = J_opt; t_ipopt_hi[k] = t_hi

    # Standard tolerance by default; known-optimum stopping only in oracle mode.
    w_bm, t_bm, J_bm = JuMP_solver("Ipopt", para, oracle_mode ? 1e-2 : 1e-6)
    push!(metrics_bm, cone_metrics(A, b, w_bm))
    J_ipopt_bm[k] = J_bm; t_ipopt_bm[k] = t_bm
    viol_bm[k] = max(maximum(A * w_bm .- b), maximum(-w_bm), abs(sum(w_bm) - 1), 0.0)

    # learned splitting ADMM (Gurobi free)
    it = Ref(0)
    sol, t_s, J_s = sLME_ADMM(para, mgrad, (args...) -> (it[] = args[6]; (!oracle_mode || sLME_ADMM_callback(args...))))
    it_slme[k] = it[]
    w = Vector{FloatType}(sol[1:n])
    push!(metrics_sl, cone_metrics(A, b, w))
    J_slme[k] = metrics_sl[end].obj
    t_slme[k]  = t_s
    eq_slme[k] = abs(sum(w) - 1)
    in_slme[k] = max(maximum(A * w .- b), maximum(-w), 0.0)
    W_slme[k, :] = w
    if k % 5 == 0; GC.gc(); end
    @printf("  [%3d/%3d] Jopt=%.6f  Ipopt_bm=%.6f (%.2f ms)  sLME=%.6f (%.2f ms, eq=%.1e, ineq=%.1e)\n",
            k, count_, J_opt, J_bm, 1e3t_bm, J_slme[k], 1e3t_s, eq_slme[k], in_slme[k])
end

result = Dict(
    "schema_version" => 2,
    "note" => "Measured original-constraint feasibility and exact objective domain. " *
              (oracle_mode ? "Oracle-assisted stopping; reference-solve cost excluded." : "Deployment stopping; no optimum oracle.") *
              " Gurobi-dependent baselines not run.",
    "n_instances" => count_, "max_opt_gap" => oracle_mode ? max_opt_gap : nothing,
    "methods" => Dict(
        "Ipopt(tol=1e-8)" => metric_summary(metrics_hi, t_ipopt_hi, J_ipopt_hi),
        (oracle_mode ? "Ipopt(oracle)" : "Ipopt(deployment)") => metric_summary(metrics_bm, t_ipopt_bm, J_ipopt_hi; oracle=oracle_mode),
        "sLME-ADMM" => merge(metric_summary(metrics_sl, t_slme, J_ipopt_hi; oracle=oracle_mode),
            Dict("iterations_mean" => mean(it_slme), "iterations_median" => median(it_slme),
                 "iterations_at_cap" => sum(it_slme .>= 1000))),
        "LME-ADMM" => Dict("note" => "not_run: needs Gurobi")))

open(joinpath(out_dir, "julia_baselines.json"), "w") do f
    JSON3.pretty(f, result)
end
npzwrite(joinpath(out_dir, "julia_baselines.npz"),
         Dict("J_ipopt_hi" => J_ipopt_hi, "t_ipopt_hi" => t_ipopt_hi,
              "J_ipopt_bm" => J_ipopt_bm, "t_ipopt_bm" => t_ipopt_bm, "viol_ipopt_bm" => viol_bm,
              "J_slme" => J_slme, "t_slme" => t_slme,
              "eq_slme" => eq_slme, "ineq_slme" => in_slme, "W_slme" => W_slme, "it_slme" => it_slme,
              "eq_hi" => [r.eq for r in metrics_hi], "ineq_hi" => [r.ineq for r in metrics_hi],
              "eq_bm" => [r.eq for r in metrics_bm], "ineq_bm" => [r.ineq for r in metrics_bm],
              "domain_valid_hi" => [r.valid for r in metrics_hi],
              "domain_valid_bm" => [r.valid for r in metrics_bm],
              "domain_valid_slme" => [r.valid for r in metrics_sl]))
println("wrote ", joinpath(out_dir, "julia_baselines.json"))
