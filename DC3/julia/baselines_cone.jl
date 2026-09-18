# Runs the repository's own cone-programming solvers on the instances exported
# by the Python benchmark, and writes julia_baselines.json next to them.
#
#   julia --project=. DC3/julia/baselines_cone.jl DC3/results/cone_programming
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

out_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "DC3", "results", "cone_programming")
out_dir = isabspath(out_dir) ? out_dir : joinpath(REPO, out_dir)
inst    = npzread(joinpath(out_dir, "test_instances.npz"))

Ainst = inst["A"]                      # (count, m, n)
binst = inst["b"]                      # (count, m)
count_, m_, n_ = size(Ainst)

const n::Int = n_
const m::Int = m_
const max_opt_gap::FloatType = length(ARGS) >= 2 ? parse(FloatType, ARGS[2]) : 0.01
# benchmarkOG.jl uses `max(div(n, nthreads())+1, 50)`.  That formula exceeds `n`
# when Julia runs single-threaded (or when n is small), and examples/cone_programming/
# utils.jl::mini_batch then indexes out of bounds - so it is additionally clamped to n.
# Start Julia with `--threads=auto` to reproduce the multi-threaded setting.
const s_mb::Int = min(max(div(n, max(Threads.nthreads(), 1)) + 1, 50), n)

GUROBI_ENV = nothing                   # skip Gurobi.Env() (no license here)
include(joinpath(REPO, "examples", "cone_programming", "maxEntropy.jl"))
include(joinpath(REPO, "examples", "cone_programming", "preprocess.jl"))
include(joinpath(REPO, "examples", "cone_programming", "JuMPsolver.jl"))
include(joinpath(REPO, "examples", "cone_programming", "LME-ADMM.jl"))

@printf("cone baselines: %d instances, n=%d, m=%d, max_opt_gap=%.4g%%\n", count_, n, m, max_opt_gap)

mgrad = gradient_struct(model, s_mb, 1)

J_ipopt_hi = zeros(count_);  t_ipopt_hi = zeros(count_)
J_ipopt_bm = zeros(count_);  t_ipopt_bm = zeros(count_)
J_slme     = zeros(count_);  t_slme     = zeros(count_)
eq_slme    = zeros(count_);  in_slme    = zeros(count_)
W_slme     = zeros(count_, n)

for k in 1:count_
    global J_opt
    A = Matrix{FloatType}(Ainst[k, :, :])
    b = Vector{FloatType}(binst[k, :])
    para = data_opt(n, m, A, b, 1.0, x -> x * log(x))

    # ground truth (tol 1e-8), exactly as in benchmarkOG.jl
    _, t_hi, J_opt = JuMP_solver("Ipopt", para, 1e-8)
    J_ipopt_hi[k] = J_opt; t_ipopt_hi[k] = t_hi

    # Ipopt with the repo's optimality-gap callback (the benchmark setting)
    _, t_bm, J_bm = JuMP_solver("Ipopt", para, 1e-2, callback_struct())
    J_ipopt_bm[k] = J_bm; t_ipopt_bm[k] = t_bm

    # learned splitting ADMM (Gurobi free)
    sol, t_s, J_s = sLME_ADMM(para, mgrad, sLME_ADMM_callback)
    w = Vector{FloatType}(sol[1:n])
    J_slme[k]  = sum(x -> x > 0 ? x * log(x) : 0.0, w)
    t_slme[k]  = t_s
    eq_slme[k] = abs(sum(w) - 1)
    in_slme[k] = max(maximum(A * w .- b), maximum(-w), 0.0)
    W_slme[k, :] = w
    if k % 5 == 0; GC.gc(); end
    @printf("  [%3d/%3d] Jopt=%.6f  Ipopt_bm=%.6f (%.2f ms)  sLME=%.6f (%.2f ms, eq=%.1e, ineq=%.1e)\n",
            k, count_, J_opt, J_bm, 1e3t_bm, J_slme[k], 1e3t_s, eq_slme[k], in_slme[k])
end

gap(J, Jr) = 100 .* abs.(J .- Jr) ./ abs.(Jr)

result = Dict(
  "note" => "produced by DC3/julia/baselines_cone.jl on DC3's exported test instances; " *
            "Gurobi-based baselines (LME_ADMM with aux_solver_gen) were not run - no license.",
  "n_instances" => count_,
  "max_opt_gap" => max_opt_gap,
  "methods" => Dict(
    "Ipopt(tol=1e-8)" => Dict(
        "obj_mean" => mean(J_ipopt_hi), "gap_pct_mean" => 0.0, "gap_pct_max" => 0.0,
        "latency_median_ms" => 1e3median(t_ipopt_hi), "feasible_rate" => 1.0,
        "note" => "ground-truth reference of examples/cone_programming/benchmarkOG.jl"),
    "Ipopt(early-stop)" => Dict(
        "obj_mean" => mean(J_ipopt_bm),
        "gap_pct_mean" => mean(gap(J_ipopt_bm, J_ipopt_hi)),
        "gap_pct_max"  => maximum(gap(J_ipopt_bm, J_ipopt_hi)),
        "latency_median_ms" => 1e3median(t_ipopt_bm), "feasible_rate" => 1.0,
        "note" => "Ipopt stopped by Ipopt_callback_BM at $(max_opt_gap)% relative gap"),
    "sLME-ADMM" => Dict(
        "obj_mean" => mean(J_slme),
        "gap_pct_mean" => mean(gap(J_slme, J_ipopt_hi)),
        "gap_pct_max"  => maximum(gap(J_slme, J_ipopt_hi)),
        "latency_median_ms" => 1e3median(t_slme),
        "eq_max" => maximum(eq_slme), "ineq_max" => maximum(in_slme),
        "feasible_rate" => mean((eq_slme .<= 1e-4) .& (in_slme .<= 1e-4)),
        "note" => "examples/cone_programming/LME-ADMM.jl :: sLME_ADMM, stopped by sLME_ADMM_callback"),
    "LME-ADMM" => Dict("note" => "not_run: needs Gurobi for aux_solver_gen"),
  ))

open(joinpath(out_dir, "julia_baselines.json"), "w") do f
    JSON3.pretty(f, result)
end
npzwrite(joinpath(out_dir, "julia_baselines.npz"),
         Dict("J_ipopt_hi" => J_ipopt_hi, "t_ipopt_hi" => t_ipopt_hi,
              "J_ipopt_bm" => J_ipopt_bm, "t_ipopt_bm" => t_ipopt_bm,
              "J_slme" => J_slme, "t_slme" => t_slme,
              "eq_slme" => eq_slme, "ineq_slme" => in_slme, "W_slme" => W_slme))
println("wrote ", joinpath(out_dir, "julia_baselines.json"))
