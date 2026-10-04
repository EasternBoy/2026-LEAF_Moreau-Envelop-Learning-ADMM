
using Pkg
const repo_root = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(repo_root)
Pkg.instantiate()
cd(repo_root) # Existing data/model loaders use repository-relative paths.

using Printf, Random, Statistics, LinearAlgebra, SparseArrays, StaticArrays
using JSON3, NPZ, NNlib, JuMP, Distributions
using Base.Threads
import ParametricOptInterface as POI
import MathOptInterface as MOI
if Sys.isapple()
    using AppleAccelerate
end

# Optional arguments (the defaults are the constants below):
#   julia --project=. --threads=8 experiments/power_grid/table.jl [--N=96] [--gopt=0.1] [--model=<path>.json --tag=<tag>]
# --model replaces the ICNN of problems/power_grid/setup.jl; its results go to
# results/power_grid/table/<tag>/gap=<g_opt>/ instead of results/power_grid/table/gap=<g_opt>/.
function table_arg(name, default)
    i = findfirst(a -> startswith(a, "--$name="), ARGS)
    return i === nothing ? default : String(split(ARGS[i], "="; limit = 2)[2])
end

const FloatType = Float64
const tol = 1e-2
const admm_tol = 1e-2
const smel_feas_tol = 1e-4   # sMEL-ADMM returned-solution feasibility tolerance (residual: tol)
const smel_gamma = parse(Float64, table_arg("smel-gamma", "1.2"))   # sMEL-ADMM over-relaxation
const admm_rho = 1.
const smel_state_scale = 400.0
const nsamples = 1000
const g_opt = parse(Float64, table_arg("gopt", "0.1"))
const s_mb = 24
const model_tag = table_arg("tag", "")
const POWER_GRID_MODEL = table_arg("model", "models/power_grid/neco_mpc-rho=1.json")
table_arg("model", nothing) !== nothing && isempty(model_tag) &&
    error("--model needs --tag, so that its results do not overwrite the default model's")
const output_dir = joinpath(repo_root, "results", "power_grid", "table", model_tag, "gap=$(g_opt)")
const seed = 20262309
const gc_every = 1
const max_iter = 1000
const feas_tol = 1e-6
const reference_tol = 1e-10
const methods = [
    "IPOPT",
    "MadNLP",
    "ADMM",
    "MEL-ADMM",
    "sMEL-ADMM",
]

residual_tolerance(name) = name == "IPOPT" ? reference_tol : name == "ADMM" ? admm_tol : tol
feasibility_tolerance(name) = name == "sMEL-ADMM" ? smel_feas_tol : residual_tolerance(name)

# Initialize Gurobi only when a selected method needs it.
GUROBI_ENV = nothing
const power_grid_problem_dir = joinpath(repo_root, "problems", "power_grid")
include(joinpath(power_grid_problem_dir, "problem.jl"))
include(joinpath(power_grid_problem_dir, "setup.jl"))
include(joinpath(power_grid_problem_dir, "jump_solver.jl"))
include(joinpath(power_grid_problem_dir, "admm.jl"))
include(joinpath(power_grid_problem_dir, "lme_admm.jl"))

# Select the benchmark horizon without changing energy_mag() for other experiments.
N = parse(Int, table_arg("N", "96"))
const input_tag = "N=$(N)"
const power_table_input_dir = joinpath(repo_root, "results", "power_grid", "table", "instances")
const input_file = joinpath(power_table_input_dir, "test_instances_$(input_tag).npz")
let d = mpc_data
    c = eco_mpc(d.r_ec, d.r_df, d.r_op, d.η, d.dT, N, d.a)
    global mpc_data = MPCData_eco(d.A, d.B, d.r_ec, d.r_df, d.r_op, d.η,
        d.BESS, d.dT, d.a, d.x_min, d.x_max, d.x_end_min, d.u_min, d.u_max, d.x0,
        d.dim, N, d.load_forecast, d.gen_forecast, admm_rho, c)
end

const case_tag = "$(input_tag)_rho=$(admm_rho)_state_scale=$(smel_state_scale)_" *
    @sprintf("optgap=%.6g_residual_ipopt=%.0e_admm=%.0e_others=%.0e_smelfeas=%.0e_smelgamma=%.2g",
             g_opt, reference_tol, admm_tol, tol, smel_feas_tol, smel_gamma)

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
# to reference_tol; their validated times are conservative upper bounds, and
# stopping_mode in the output makes that distinction explicit.
const madnlp_gap_callback = isdefined(MadNLP, :AbstractUserCallback)
if madnlp_gap_callback
    struct TableMadNLPCallback <: MadNLP.AbstractUserCallback
        target::GapTarget
        tol::Float64
    end
    function (cb::TableMadNLPCallback)(solver, mode)
        cb.target.iterations = solver.cnt.k
        mode isa MadNLP.UserCallbackRegular || return true
        cb.target.residual = MadNLP.get_inf_pr(solver)
        cb.target.reached = gap_reached(cb.target, solver.obj_val) &&
                            cb.target.residual < cb.tol
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
    return max(residual, d.x_end_min - x)
end

function nlp_runner(name, target; reference = false)
    handle = Ref{JuMP.Model}()
    configure = function (m)
        handle[] = m
        if name == "IPOPT"
            set_optimizer_attribute(m, "max_iter", max_iter)
            if !reference
                cb = function (mode, iter, obj, inf_pr, args...)
                    target.iterations = iter
                    target.residual = inf_pr
                    target.reached = gap_reached(target, obj) &&
                                     target.residual < residual_tolerance(name)
                    return !target.reached
                end
                MOI.set(m, Ipopt.CallbackFunction(), cb)
            end
        else
            set_optimizer_attribute(m, "max_iter", max_iter)
            if madnlp_gap_callback
                set_optimizer_attribute(m, "intermediate_callback",
                                        TableMadNLPCallback(target, residual_tolerance(name)))
            end
        end
    end
    # In time mode, keep the NLP solver's native tolerance tight so it cannot
    # declare convergence before the gap-and-residual callback is satisfied.
    solver_tol = reference_tol
    solve = mpc_eco_solver(name, mpc_data, solver_tol; configure)
    mode = reference ? "reference" : name == "MadNLP" && !madnlp_gap_callback ?
        "residual_tolerance_upper_bound" : "gap_and_residual"
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
    solver_tol = residual_tolerance(name)
    if name == "ADMM"
        solve = ADMM_eco_iter(d, prime_sol_struct("Ipopt", d), aux_solver_eco("Gurobi", d);
                             tol = solver_tol, max_iter = max_iter)
    elseif name == "MEL-ADMM"
        solve = LME_ADMM(d, gradient_struct(model, s_mb, dim; kernel = mmul_add_matrix!),
                         aux_solver_eco("Gurobi", d))
    else
        solve = LME_ADMM_split(d, gradient_struct(model, s_mb, dim; kernel = mmul_add_matrix!),
                               dynamics_projection(d; state_scale=smel_state_scale))
    end
    cb = function (z, w, α, iter, J, seconds)
        target.iterations = iter
        target.residual = maximum(abs, w .- z)
        target.reached = gap_reached(target, J)
        return target.reached
    end
    split_cb = function (z, w, α, v, β, iter, J)
        target.iterations = iter
        target.residual = max(maximum(abs, w .- v), maximum(abs, v .- z))
        target.reached = gap_reached(target, J)
        return target.reached
    end
    return function (x0, load, gen)
        target.iterations = 0
        target.reached = false
        result = if name == "ADMM"
            solve(x0, load, gen, cb)
        elseif name == "MEL-ADMM"
            solve(x0, load, gen, cb; tol = solver_tol, max_iter = max_iter)
        else
            solve(x0, load, gen, split_cb; tol = solver_tol, max_iter = max_iter,
                  feas_tol = feasibility_tolerance(name), γ = smel_gamma)
        end
        if name in ("MEL-ADMM", "sMEL-ADMM")
            target.reached = target.reached && target.residual < solver_tol &&
                eco_solution_feasible(d, result[1], x0, load, gen, feasibility_tolerance(name))
        end
        stopping_mode = if name in ("MEL-ADMM", "sMEL-ADMM")
            "gap_residual_and_feasibility"
        else
            "gap_callback_with_solver_termination"
        end
        return (; vars = result[1], seconds = result[2], iterations = target.iterations,
                status = target.reached ? "converged" : "iteration_limit",
                stopping_mode,
                stopping_residual = target.residual)
    end
end

function result_row(name, target, run, sample, x0, load, gen)
    J = get_objective(mpc_data, run.vars)
    gap = gap_percent(J, target.reference)
    residual = feasibility(run.vars, x0, load, gen)
    return (; sample, x0, solver = name, target_gap_pct = target.percent,
        residual_tol = residual_tolerance(name),
        solve_time_ms = 1000 * run.seconds, objective = J, reference_objective = target.reference,
        opt_gap_pct = gap, feasibility_residual = residual, stopping_residual = run.stopping_residual,
        iterations = run.iterations, status = run.status, stopping_mode = run.stopping_mode)
end

function main()
    nsamples >= 2 || error("Use at least two measured samples for mean ± standard deviation.")
    isfinite(g_opt) && g_opt > 0 || error("g_opt must be a positive percentage.")
    all(m -> m in ("IPOPT", "MadNLP", "ADMM", "MEL-ADMM", "sMEL-ADMM"), methods) || error("Unknown methods entry.")
    isfile(input_file) || error(
        "Missing input file $input_file; run `python3 " *
        "experiments/power_grid/generate_table_instances.py --N $N` from $repo_root."
    )
    if "MEL-ADMM" in methods && nthreads() < 2
        error("Launch Julia with --threads=8 (or another count > 1) for threaded MEL mini-batches.")
    end
    mkpath(output_dir)
    files = [joinpath(output_dir, "$(stem)_$(case_tag).$(ext)") for (stem, ext) in
             (("ipopt_references", "npz"), ("results", "csv"), ("summary", "csv"),
              ("metadata", "json"), ("table", "md"))]

    instances = npzread(input_file)
    Int(only(instances["N"])) == N || error("Input horizon does not match N=$N.")
    Int(only(instances["seed"])) == seed || error("Input seed does not match seed=$seed.")
    x0s = Vector{Float64}(instances["x0"])
    loads = Matrix{Float64}(instances["load"])
    gens = Matrix{Float64}(instances["gen"])
    length(x0s) == nsamples || error("Input sample count does not match nsamples=$nsamples.")
    size(loads) == (nsamples, N) && size(gens) == size(loads) || error("Input forecast shapes are invalid.")
    all(isfinite, x0s) && all(isfinite, loads) && all(isfinite, gens) || error("Input contains nonfinite values.")
    all(loads .== loads[1:1, :]) && all(gens .== gens[1:1, :]) ||
        error("This benchmark expects fixed load and generation forecasts across instances.")
    load = vec(loads[1, :])
    gen = vec(gens[1, :])

    target = GapTarget()
    reference = nlp_runner("IPOPT", target; reference = true)
    reference(0.5, load, gen) # First reference run excluded.
    references = zeros(nsamples)
    for k in 1:nsamples
        run = reference(x0s[k], load, gen)
        feasibility(run.vars, x0s[k], load, gen) <= feas_tol || error("Infeasible reference at sample $k")
        references[k] = get_objective(mpc_data, run.vars)
        isfinite(references[k]) || error("Nonfinite reference at sample $k")
        k % gc_every == 0 && GC.gc()
    end
    instances["J_opt"] = references
    npzwrite(files[1], instances)

    if any(m -> m in ("ADMM", "MEL-ADMM"), methods)
        global GUROBI_ENV = Gurobi.Env()
    end
    runners = [name in ("IPOPT", "MadNLP") ? nlp_runner(name, target) : admm_runner(name, target) for name in methods]
    metadata = Dict("N" => N, "rho" => admm_rho, "primary_scalar_variables" => dim * N,
        "smel_state_scale" => smel_state_scale, "icnn_model" => POWER_GRID_MODEL,
        "case_tag" => case_tag, "source_instances_file" => input_file,
        "reference_instances_file" => basename(files[1]),
        "results_file" => basename(files[2]), "summary_file" => basename(files[3]),
        "table_file" => basename(files[5]),
        "measured_samples" => nsamples, "g_opt_pct" => g_opt,
        "residual_tolerances" => Dict(name => residual_tolerance(name) for name in methods),
        "feasibility_tolerances" => Dict(name => feasibility_tolerance(name) for name in methods),
        "smel_gamma" => smel_gamma,
        "seed" => seed,
        "x0_distribution" => "See source instance file generated by " *
            "experiments/power_grid/generate_table_instances.py",
        "methods" => methods,
        "julia_threads" => nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "julia_version" => string(VERSION), "gc_every" => gc_every,
        "max_iter" => max_iter, "feasibility_tolerance" => feas_tol,
        "reference_tolerance" => reference_tol, "discarded_warmups_per_solver_per_gap" => 1,
        "timing" => "Existing solver-returned seconds; split solver uses internal elapsed time",
        "summary_std" => "Sample standard deviation across all measured instances, not standard error",
        "madnlp_gap_callback" => madnlp_gap_callback,
        "DC3" => "Not run; use $input_file with CLARABEL as its reference solver")
    open(files[4], "w") do io
        JSON3.pretty(io, metadata)
    end

    rows = NamedTuple[]
    target.percent = g_opt
    target.reference = references[1]
    for (name, runner) in zip(methods, runners)
        run = runner(x0s[1], load, gen)
        result_row(name, target, run, 0, x0s[1], load, gen)
    end
    GC.gc()
    for k in 1:nsamples
        target.reference = references[k]
        for (name, runner) in zip(methods, runners)
            run = runner(x0s[k], load, gen)
            row = result_row(name, target, run, k, x0s[k], load, gen)
            push!(rows, row)
            # Save every result, including iteration-limited runs.
            CSV.write(files[2], DataFrame([row]); append = length(rows) > 1)
        end
        k % gc_every == 0 && GC.gc()
        if k % gc_every == 0 || k == nsamples
            @printf("%d/%d measured instances complete\n", k, nsamples)
        end
    end

    summaries = NamedTuple[]
    for name in methods
        group = filter(r -> r.solver == name, rows)
        times = [r.solve_time_ms for r in group]
        μ = isempty(times) ? NaN : mean(times)
        σ = length(times) < 2 ? NaN : std(times)
        gaps = [r.opt_gap_pct for r in group]
        violations = [r.feasibility_residual for r in group]
        push!(summaries, (; solver = name, target_gap_pct = g_opt,
            residual_tol = residual_tolerance(name),
            time_mean_ms = μ, time_std_ms = σ, time_max_ms = maximum(times), samples = length(group),
            opt_gap_mean_pct = mean(gaps), opt_gap_std_pct = std(gaps), opt_gap_max_pct = maximum(gaps),
            constr_viol_mean = mean(violations), constr_viol_max = maximum(violations)))
    end
    CSV.write(files[3], DataFrame(summaries))
    render_results(summaries, files[5])
    println("Saved raw results, summary, Markdown table, metadata, and IPOPT references in ", output_dir)
    foreach(f -> println("  ", basename(f)), files)
    println("DC3: not run. Summaries include all measured runs.")
end

function render_results(summaries, path)
    best = argmin([s.time_mean_ms for s in summaries])
    rows = String[]
    for (i, s) in enumerate(summaries)
        timing = @sprintf("%.3f (%.3f)", s.time_mean_ms, s.time_max_ms)
        i == best && (timing = "**$timing**")
        gap = @sprintf("%.6g ± %.6g (%.6g)",
                       s.opt_gap_mean_pct, s.opt_gap_std_pct, s.opt_gap_max_pct)
        violation = @sprintf("%.1e (%.1e)", s.constr_viol_mean, s.constr_viol_max)
        push!(rows, @sprintf("| %s | %.1e | %s | %s | %s |",
                            s.solver, s.residual_tol, timing, gap, violation))
    end
    condition = "`g_opt ≤ $(g_opt)%` plus each solver's native checks; " *
        "MEL-ADMM and sMEL-ADMM also require their residual tolerance and returned-solution feasibility"
    md = """
# Power-grid benchmark: solving time and optimality gap

Generated by `experiments/power_grid/table.jl` from `$input_file`.
The stopping condition is $condition. Results use $nsamples measured instances;
solving time is in ms, shown as mean (maximum), and optimality gap is in %, shown
as mean ± sample standard deviation (maximum). Constraint violation is the largest
violation of the original power-grid constraints and is shown as mean (maximum).
**Bold** marks the lowest mean solving time. Warm-up runs and IPOPT reference-solve
costs are excluded.

| Solver | Residual tol | Solving time mean (max) | Opt. gap mean ± std (max) | Constr. viol. mean (max) |
|---|---:|---:|---:|---:|
$(join(rows, "\n"))
"""
    write(path, md)
    println(md)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
