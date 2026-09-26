
using Pkg
Pkg.activate(".")
Pkg.instantiate()
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO_ROOT)
cd(REPO_ROOT) # Existing data/model loaders use repository-relative paths.

using Printf, Random, Statistics, LinearAlgebra, SparseArrays, StaticArrays
using JSON3, NPZ, NNlib, JuMP, Distributions
using Base.Threads
import ParametricOptInterface as POI
import MathOptInterface as MOI
if Sys.isapple()
    using AppleAccelerate
end

const FloatType = Float64
const BENCHMARK_MODE = :time # Change to :optimality_gap for the table's gap column.
const tol = 1e-4
const ADMM_TOL = 1e-2
const s_mb = 24
const NSAMPLES = 1000
const G_OPT = 0.01
const OUTPUT_DIR = joinpath(REPO_ROOT, "data", "solving_data",
    BENCHMARK_MODE == :time ? "power_table_time_gap=$(G_OPT)" : "power_table_opt_gap")
const SEED = 20262309
const GC_EVERY = 1 # Same cadence as power_grid/get_benchmark.
const MAX_ITER = 1000
const FEAS_TOL = 1e-6
const REFERENCE_TOL = 1e-10
const METHODS = ["IPOPT", "MadNLP", "ADMM", "MEL-ADMM", "sMEL-ADMM"]

residual_tolerance(name) = name == "IPOPT" ? REFERENCE_TOL : name == "ADMM" ? ADMM_TOL : tol

# Initialize Gurobi only when a selected method needs it.
GUROBI_ENV = nothing
include("power_system.jl")
include("preprocess.jl")
include("eMPC_JuMPsolver.jl")
include("eMPC_ADMM.jl")
include("eMPC_L-ADMM.jl")

# Select the benchmark horizon without changing energy_mag() for other experiments.
N = 96
const INPUT_TAG = "N=$(N)"
const INPUT_FILE = joinpath(REPO_ROOT, "data", "solving_data", "power_table_inputs",
    "test_instances_$(INPUT_TAG).npz")
let d = mpc_data
    c = eco_mpc(d.r_ec, d.r_df, d.r_op, d.η, d.dT, N, d.a)
    global mpc_data = MPCData_eco(d.A, d.B, d.r_ec, d.r_df, d.r_op, d.η,
        d.BESS, d.dT, d.a, d.x_min, d.x_max, d.u_min, d.u_max, d.x0,
        d.dim, N, d.load_forecast, d.gen_forecast, d.rho, c)
end

const CASE_TAG = "$(INPUT_TAG)_" * (BENCHMARK_MODE == :time ? "optgap=$(G_OPT)" :
    @sprintf("residual_ipopt=%.0e_admm=%.0e_others=%.0e", REFERENCE_TOL, ADMM_TOL, tol))

gap_percent(J, Jref) = 100 * abs(J - Jref) / max(abs(Jref), eps(Float64))

@kwdef mutable struct GapTarget
    reference::Float64 = NaN
    percent::Float64 = 0.01
    iterations::Int = 0
    residual::Float64 = NaN
    reached::Bool = false
end

function gap_reached(target, J)
    return isfinite(J) && gap_percent(J, target.reference) <= target.percent
end

# Newer MadNLP versions expose an intermediate callback. Older versions solve
# to REFERENCE_TOL; their validated times are conservative upper bounds, and
# stopping_mode in the output makes that distinction explicit.
const MADNLP_GAP_CALLBACK = isdefined(MadNLP, :AbstractUserCallback)
if MADNLP_GAP_CALLBACK
    struct TableMadNLPCallback <: MadNLP.AbstractUserCallback
        target::GapTarget
    end
    function (cb::TableMadNLPCallback)(solver, mode)
        cb.target.iterations = solver.cnt.k
        mode isa MadNLP.UserCallbackRegular || return true
        cb.target.reached = gap_reached(cb.target, solver.obj_val)
        return !cb.target.reached
    end
end

# Check constraints of the original problem, reconstructing states from u.
function feasibility(vars, x0, load, gen)
    d = mpc_data
    x = x0
    residual = 0.0
    for k in 1:d.N
        m, u, p = vars[1, k], vars[2, k], vars[3, k]
        x = d.A * x + d.B * u
        residual = max(residual, abs(u + m + gen[k] - load[k] - p),
            d.u_min - u, u - d.u_max, -p, d.x_min - x, x - d.x_max)
        size(vars, 1) == 4 && (residual = max(residual, abs(vars[4, k] - x)))
    end
    return max(residual, abs(x - x0))
end

function nlp_runner(name, target; reference = false)
    handle = Ref{JuMP.Model}()
    configure = function (m)
        handle[] = m
        if name == "IPOPT"
            set_optimizer_attribute(m, "max_iter", MAX_ITER)
            if !reference && BENCHMARK_MODE == :time
                cb = function (mode, iter, obj, inf_pr, args...)
                    target.iterations = iter
                    target.reached = gap_reached(target, obj)
                    return !target.reached
                end
                MOI.set(m, Ipopt.CallbackFunction(), cb)
            end
        else
            set_optimizer_attribute(m, "max_iter", MAX_ITER)
            if BENCHMARK_MODE == :time && MADNLP_GAP_CALLBACK
                set_optimizer_attribute(m, "intermediate_callback", TableMadNLPCallback(target))
            end
        end
    end
    solver_tol = reference || BENCHMARK_MODE == :time ? REFERENCE_TOL : residual_tolerance(name)
    solve = mpc_eco_solver(name, mpc_data, solver_tol; configure)
    mode = reference ? "reference" : BENCHMARK_MODE == :optimality_gap ?
        "residual_tolerance" : name == "MadNLP" && !MADNLP_GAP_CALLBACK ?
        "tight_tolerance_upper_bound" : "gap_callback"
    return function (x0, load, gen)
        target.iterations = 0
        target.reached = false
        vars, seconds, _ = solve(x0, load, gen)
        status = termination_status(handle[])
        if reference && !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
            error("IPOPT reference failed at x0=$x0: $status")
        end
        iterations = barrier_iterations(handle[])
        return (; vars, seconds, iterations, status = string(status), stopping_mode = mode,
                stopping_residual = NaN)
    end
end

function admm_runner(name, target)
    d = mpc_data
    solver_tol = BENCHMARK_MODE == :time ? -Inf : residual_tolerance(name)
    if name == "ADMM"
        solve = ADMM_eco_iter(d, prime_sol_struct("Ipopt", d), aux_solver_eco("Gurobi", d);
                             tol = solver_tol, max_iter = MAX_ITER)
    elseif name == "MEL-ADMM"
        solve = LME_ADMM(d, gradient_struct(model, s_mb, dim), aux_solver_eco("Gurobi", d))
    else
        solve = LME_ADMM_split(d, gradient_struct(model, s_mb, dim), dynamics_projection(d))
    end
    cb = function (z, w, α, iter, J, seconds)
        target.iterations = iter
        target.residual = maximum(w - z)
        target.reached = BENCHMARK_MODE == :time ? gap_percent(J, target.reference) <= target.percent :
                         target.residual < solver_tol
        return target.reached
    end
    split_cb = function (z, w, α, v, β, iter, J)
        target.iterations = iter
        target.residual = max(maximum(w - v), maximum(v - z))
        target.reached = BENCHMARK_MODE == :time ?
            gap_percent(J, target.reference) <= target.percent : target.residual < solver_tol
        return target.reached
    end
    return function (x0, load, gen)
        target.iterations = 0
        target.reached = false
        # Residual-only exits are disabled only in the timing experiment.
        result = if name == "ADMM"
            solve(x0, load, gen, cb)
        elseif name == "MEL-ADMM"
            solve(x0, load, gen, cb; tol = solver_tol, max_iter = MAX_ITER)
        else
            solve(x0, load, gen, split_cb; tol = solver_tol, max_iter = MAX_ITER)
        end
        return (; vars = result[1], seconds = result[2], iterations = target.iterations,
                status = target.reached ? "converged" : "iteration_limit",
                stopping_mode = BENCHMARK_MODE == :time ? "gap_callback" : "residual_tolerance",
                stopping_residual = target.residual)
    end
end

function result_row(name, target, run, sample, x0, load, gen)
    J = get_objective(mpc_data, run.vars)
    gap = gap_percent(J, target.reference)
    residual = feasibility(run.vars, x0, load, gen)
    return (; sample, x0, solver = name, benchmark_mode = string(BENCHMARK_MODE),
        target_gap_pct = BENCHMARK_MODE == :time ? target.percent : NaN,
        residual_tol = BENCHMARK_MODE == :optimality_gap ? residual_tolerance(name) : NaN,
        solve_time_ms = 1000 * run.seconds, objective = J, reference_objective = target.reference,
        opt_gap_pct = gap, feasibility_residual = residual, stopping_residual = run.stopping_residual,
        iterations = run.iterations, status = run.status, stopping_mode = run.stopping_mode)
end

function main()
    NSAMPLES >= 2 || error("Use at least two measured samples for mean ± standard deviation.")
    BENCHMARK_MODE in (:time, :optimality_gap) || error("Choose :time or :optimality_gap.")
    isfinite(G_OPT) && G_OPT > 0 || error("G_OPT must be a positive percentage.")
    all(m -> m in ("IPOPT", "MadNLP", "ADMM", "MEL-ADMM", "sMEL-ADMM"), METHODS) || error("Unknown METHODS entry.")
    isfile(INPUT_FILE) || error("Missing input file $INPUT_FILE; run DC3/benchmark/generate_table_instances.py first.")
    if any(m -> m in ("MEL-ADMM", "sMEL-ADMM"), METHODS) && nthreads() < 2
        error("Launch Julia with --threads=8 (or another count > 1) for threaded MEL mini-batches.")
    end
    mkpath(OUTPUT_DIR)
    # Prevent accidental replacement of an earlier scientific experiment.
    files = [joinpath(OUTPUT_DIR, "$(stem)_$(CASE_TAG).$(ext)") for (stem, ext) in
             (("ipopt_references", "npz"), ("results", "csv"), ("summary", "csv"), ("metadata", "json"))]
    any(isfile, files) && error("Output files already exist; choose a new output directory.")

    instances = npzread(INPUT_FILE)
    Int(only(instances["N"])) == N || error("Input horizon does not match N=$N.")
    Int(only(instances["seed"])) == SEED || error("Input seed does not match SEED=$SEED.")
    x0s = Vector{Float64}(instances["x0"])
    loads = Matrix{Float64}(instances["load"])
    gens = Matrix{Float64}(instances["gen"])
    length(x0s) == NSAMPLES || error("Input sample count does not match NSAMPLES=$NSAMPLES.")
    size(loads) == (NSAMPLES, N) && size(gens) == size(loads) || error("Input forecast shapes are invalid.")
    all(isfinite, x0s) && all(isfinite, loads) && all(isfinite, gens) || error("Input contains nonfinite values.")
    all(loads .== loads[1:1, :]) && all(gens .== gens[1:1, :]) ||
        error("This benchmark expects fixed load and generation forecasts across instances.")
    load = vec(loads[1, :])
    gen = vec(gens[1, :])

    target = GapTarget()
    reference = nlp_runner("IPOPT", target; reference = true)
    reference(0.5, load, gen) # First reference run excluded.
    references = zeros(NSAMPLES)
    for k in 1:NSAMPLES
        run = reference(x0s[k], load, gen)
        feasibility(run.vars, x0s[k], load, gen) <= FEAS_TOL || error("Infeasible reference at sample $k")
        references[k] = get_objective(mpc_data, run.vars)
        isfinite(references[k]) || error("Nonfinite reference at sample $k")
        k % GC_EVERY == 0 && GC.gc()
    end
    instances["J_opt"] = references
    npzwrite(files[1], instances)

    if any(m -> m in ("ADMM", "MEL-ADMM"), METHODS)
        global GUROBI_ENV = Gurobi.Env()
    end
    runners = [name in ("IPOPT", "MadNLP") ? nlp_runner(name, target) : admm_runner(name, target) for name in METHODS]
    metadata = Dict("N" => N, "primary_scalar_variables" => dim * N,
        "case_tag" => CASE_TAG, "source_instances_file" => INPUT_FILE,
        "reference_instances_file" => basename(files[1]),
        "results_file" => basename(files[2]), "summary_file" => basename(files[3]),
        "measured_samples" => NSAMPLES, "benchmark_mode" => string(BENCHMARK_MODE),
        "g_opt_pct" => BENCHMARK_MODE == :time ? G_OPT : nothing,
        "residual_tolerances" => BENCHMARK_MODE == :optimality_gap ?
            Dict(name => residual_tolerance(name) for name in METHODS) : nothing, "seed" => SEED,
        "x0_distribution" => "See source instance file generated by DC3/benchmark/generate_table_instances.py",
        "methods" => METHODS,
        "julia_threads" => nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "julia_version" => string(VERSION), "gc_every" => GC_EVERY,
        "max_iter" => MAX_ITER, "feasibility_tolerance" => FEAS_TOL,
        "reference_tolerance" => REFERENCE_TOL, "discarded_warmups_per_solver_per_gap" => 1,
        "timing" => "Existing solver-returned seconds; split solver uses internal elapsed time",
        "summary_std" => "Sample standard deviation across all measured instances, not standard error",
        "madnlp_gap_callback" => MADNLP_GAP_CALLBACK,
        "DC3" => "Not run; use $INPUT_FILE with CLARABEL as its reference solver")
    open(files[4], "w") do io
        JSON3.pretty(io, metadata)
    end

    rows = NamedTuple[]
    target.percent = G_OPT
    target.reference = references[1]
    for (name, runner) in zip(METHODS, runners)
        # Same call path and result checks as measured runs, discarded.
        run = runner(x0s[1], load, gen)
        result_row(name, target, run, 0, x0s[1], load, gen)
    end
    GC.gc()
    for k in 1:NSAMPLES
        target.reference = references[k]
        for (name, runner) in zip(METHODS, runners)
            run = runner(x0s[k], load, gen)
            row = result_row(name, target, run, k, x0s[k], load, gen)
            push!(rows, row)
            # Save every result, including iteration-limited runs.
            CSV.write(files[2], DataFrame([row]); append = length(rows) > 1)
        end
        k % GC_EVERY == 0 && GC.gc()
        if k % GC_EVERY == 0 || k == NSAMPLES
            @printf("%s: %d/%d measured instances complete\n", string(BENCHMARK_MODE), k, NSAMPLES)
        end
    end

    summaries = NamedTuple[]
    for name in METHODS
        group = filter(r -> r.solver == name, rows)
        times = [r.solve_time_ms for r in group]
        μ = isempty(times) ? NaN : mean(times)
        σ = length(times) < 2 ? NaN : std(times)
        gaps = [r.opt_gap_pct for r in group]
        push!(summaries, (; solver = name, benchmark_mode = string(BENCHMARK_MODE),
            target_gap_pct = BENCHMARK_MODE == :time ? G_OPT : NaN,
            residual_tol = BENCHMARK_MODE == :optimality_gap ? residual_tolerance(name) : NaN,
            time_mean_ms = μ, time_std_ms = σ, samples = length(group),
            opt_gap_mean_pct = mean(gaps), opt_gap_std_pct = std(gaps), opt_gap_max_pct = maximum(gaps)))
    end
    CSV.write(files[3], DataFrame(summaries))
    print_results_table(summaries)
    println("\nSaved raw results, summary, metadata, and IPOPT references in ", OUTPUT_DIR)
    foreach(f -> println("  ", basename(f)), files)
    println("DC3: not run. Summaries include all measured runs.")
end

function print_results_table(summaries)
    println("\nResults: N=$N, $NSAMPLES measured instances; mean ± sample standard deviation (warm-ups excluded)")
    if BENCHMARK_MODE == :time
        println("Timing target: g_opt ≤ $(G_OPT)%")
        println("| Solver     | Solve time (ms)           |")
        println("|------------|---------------------------|")
        for s in summaries
            @printf("| %-10s | %10.3f ± %-10.3f |\n",
                    s.solver, s.time_mean_ms, s.time_std_ms)
        end
    else
        println("| Solver     | Residual tol | Mean gap ± std (%)        | Max gap (%)  | Solve time (ms)           |")
        println("|------------|--------------|---------------------------|--------------|---------------------------|")
        for s in summaries
            @printf("| %-10s | %12.1e | %10.6g ± %-10.6g | %12.6g | %10.3f ± %-10.3f |\n",
                    s.solver, s.residual_tol, s.opt_gap_mean_pct, s.opt_gap_std_pct,
                    s.opt_gap_max_pct, s.time_mean_ms, s.time_std_ms)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
