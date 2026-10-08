# QP + L1 benchmark: OSQP (slack form), sLME-ADMM with two ICNN checkpoints, DC3 + correction.
#   julia --project=. experiments/qp_l1/table.jl            # run missing Julia results, then render
#   julia --project=. experiments/qp_l1/table.jl --force    # rerun the Julia solvers
#   julia --project=. experiments/qp_l1/table.jl --render   # render saved results only
# DC3 + correction results come from experiments/qp_l1/table.py.
const REPO = abspath(joinpath(@__DIR__, "..", ".."))
using Pkg
Base.active_project() == joinpath(REPO, "Project.toml") || Pkg.activate(REPO; io = devnull)
using Printf, Random, SparseArrays, LinearAlgebra, NPZ, JuMP, Statistics
using Distributions

const OUT = joinpath(REPO, "results", "qp_l1", "table")
cd(REPO)

const FloatType = Float64
const n = 100
const neq = 50
const m = 50
const N_SAMPLES = 1000
const SEED = 20260923            # the QP benchmark seed: the same x as results/qp/table
const OSQP_TOL = 1e-8
const SLME_TOL = 1e-2
const SLME_FEAS_TOL = 1e-6
const MAX_ITER = 1000
const GAPS = (1.0, 11.0)         # objective-gap targets (%)
const C_V = 1e-4                 # feasibility threshold for a timing cell
const ZERO_TOL = 1e-4            # |y_i| ≤ ZERO_TOL counts as zero
const MODELS = ["5000ep" => "models/qp_l1/qp_l1-rho=1-128x128-5000ep.npz",
                "10000ep" => "models/qp_l1/qp_l1-rho=1-128x128-10000ep.npz"]

include(joinpath(REPO, "problems", "qp_l1", "problem.jl"))
include(joinpath(REPO, "problems", "qp_l1", "setup.jl"))
include(joinpath(REPO, "problems", "qp_l1", "jump_solver.jl"))

const MATRICES = ("Q", "p", "A", "G", "h")
slme_file(tag, gap) = "sLME-ADMM-$(tag)-gopt=$(gap).npz"

function instances()
    path = joinpath(OUT, "instances.npz")
    if isfile(path)
        saved = npzread(path)
        @assert all(saved[key] ≈ qp_data[key] for key in MATRICES)
        return saved["X"]
    end
    Random.seed!(SEED)
    data_opt(n, neq, m) # Skip the warmup draw, matching experiments/qp/table.jl.
    X = reduce(vcat, [permutedims(data_opt(n, neq, m).x) for _ in 1:N_SAMPLES])
    mkpath(OUT)
    npzwrite(path, Dict("X" => X, "seed" => SEED, "lambda" => QP_L1_LAMBDA,
                       (key => qp_data[key] for key in MATRICES)...))
    return X
end

instance(X, k) = data_opt(; x = Vector{Float64}(X[k, :]))

function score(X, W, Jopt)
    eq = abs.(W * qp_data["A"]' - X)
    ineq = max.(W * qp_data["G"]' .- qp_data["h"]', 0.0)
    J = [all(isfinite, W[k, :]) ? get_objective(instance(X, k), Vector{Float64}(W[k, :])) : NaN
         for k in 1:size(X, 1)]
    max_eq = vec(maximum(eq; dims = 2))
    max_ineq = vec(maximum(ineq; dims = 2))
    return Dict("objective" => J,
                "gap_pct" => 100 .* abs.(J - Jopt) ./ max.(abs.(Jopt), eps(Float64)),
                "max_viol" => max.(max_eq, max_ineq),
                "zero_pct" => 100 .* vec(mean(abs.(W) .<= ZERO_TOL; dims = 2)))
end

function run_julia(X)
    mkpath(OUT)
    Wopt = zeros(N_SAMPLES, n); Topt = zeros(N_SAMPLES); Jopt = zeros(N_SAMPLES)
    JuMP_solver("osqp", instance(X, 1), OSQP_TOL) # compile
    for k in 1:N_SAMPLES
        Wopt[k, :], Topt[k], Jopt[k] = JuMP_solver("osqp", instance(X, k), OSQP_TOL)
        k % 200 == 0 && println("OSQP: $k / $N_SAMPLES")
    end
    npzwrite(joinpath(OUT, "OSQP.npz"),
             Dict("W" => Wopt, "J_opt" => Jopt, "time_ms" => 1e3 .* Topt, "tol" => OSQP_TOL,
                  "max_iter" => 100_000,
                  "seed" => SEED, "samples" => N_SAMPLES, "lambda" => QP_L1_LAMBDA))
    for (tag, path) in MODELS
        _, mp = load_model(path)
        gradient = gradient_struct(ICNN(mp), 1, n)
        for gap in GAPS
            GAP_TARGET[] = gap
            global J_opt = Jopt[1]
            sLME_ADMM(instance(X, 1), gradient, sLME_ADMM_callback; tol = SLME_TOL,
                      feas_tol = SLME_FEAS_TOL, max_iter = MAX_ITER) # compile
            W = zeros(N_SAMPLES, n); T = zeros(N_SAMPLES)
            for k in 1:N_SAMPLES
                global J_opt = Jopt[k]
                W[k, :], T[k], _ = sLME_ADMM(instance(X, k), gradient, sLME_ADMM_callback; tol = SLME_TOL,
                                             feas_tol = SLME_FEAS_TOL, max_iter = MAX_ITER)
                k % 5 == 0 && GC.gc()
            end
            println("sLME-ADMM $tag, g_opt ≤ $gap%: done")
            npzwrite(joinpath(OUT, slme_file(tag, gap)),
                     Dict("W" => W, "time_ms" => 1e3 .* T, "model_epochs" => parse(Int, chop(tag, tail = 2)), "max_opt_gap" => gap,
                          "tol" => SLME_TOL, "feas_tol" => SLME_FEAS_TOL, "max_iter" => MAX_ITER,
                          "oracle_assisted" => true, "projection_setup_excluded" => true,
                          "seed" => SEED, "samples" => N_SAMPLES, "lambda" => QP_L1_LAMBDA))
        end
    end
end

fmt(x, f) = all(isfinite, x) ? Printf.format(Printf.Format("$f ($f)"), mean(x), maximum(x)) : "undefined (non-finite)"

function render(X)
    load(file) = isfile(joinpath(OUT, file)) ? npzread(joinpath(OUT, file)) : nothing
    osqp = load("OSQP.npz")
    Jopt = osqp["J_opt"]
    # Per column: the result used for each gap target (sLME-ADMM is rerun per target).
    columns = [("OSQP (slack form)", Dict(gap => osqp for gap in GAPS))]
    for (tag, _) in MODELS
        push!(columns, ("sLME-ADMM $tag", Dict(gap => load(slme_file(tag, gap)) for gap in GAPS)))
    end
    dc3 = load("DC3.npz")
    push!(columns, ("DC3 + correction", Dict(gap => dc3 for gap in GAPS)))
    for (_, by_gap) in columns, saved in values(by_gap)
        saved === nothing || haskey(saved, "gap_pct") || merge!(saved, score(X, saved["W"], Jopt))
    end

    rows = String[]
    for gap in GAPS
        cells = map(columns) do (_, by_gap)
            saved = by_gap[gap]
            saved === nothing && return ("—", Inf)
            ok = all(saved["max_viol"] .<= C_V) && all(saved["gap_pct"] .<= gap)
            ok ? (fmt(saved["time_ms"], "%.2f"), mean(saved["time_ms"])) : ("unable to achieve", Inf)
        end
        best = argmin(last.(cells))
        text = [isfinite(c[2]) && i == best ? "**$(c[1])**" : c[1] for (i, c) in enumerate(cells)]
        push!(rows, "| Solving time (g_opt ≤ $(gap)%) | $(join(text, " | ")) |")
    end
    final = [by_gap[first(GAPS)] for (_, by_gap) in columns]
    cell(f) = join([s === nothing ? "—" : f(s) for s in final], " | ")
    push!(rows, "| Constr. viol. | $(cell(s -> fmt(s["max_viol"], "%.1e"))) |")
    push!(rows, "| Opt. gap (%) | $(cell(s -> fmt(s["gap_pct"], "%.3g"))) |")
    push!(rows, "| Zeros (%) | $(cell(s -> fmt(s["zero_pct"], "%.1f"))) |")
    header = join([name * " mean (max)" for (name, _) in columns], " | ")
    md = """
    # QP + L1: n = $n, neq = $neq, m = $m, λ = $QP_L1_LAMBDA

    min ½y'Qy + p'y + λ‖y‖₁ s.t. Ay = x, Gy ≤ h, with dense Q (problems/qp_l1/problem.jl),
    over $(size(X, 1)) instances (seed $SEED, the x of the QP benchmark). Time in ms, gap in %,
    as mean (maximum). Constraint violation is the maximum equality/inequality violation per
    instance. Zeros: share of components with |y_i| ≤ $ZERO_TOL.

    OSQP solves the slack form in (y, t) with -t ≤ y ≤ t (200 variables) at tolerance $OSQP_TOL
    and is the reference optimum. sLME-ADMM works on y directly with the learned Moreau envelope
    (consensus tolerance $SLME_TOL, feasibility tolerance $SLME_FEAS_TOL, at most $MAX_ITER
    iterations) and is rerun for each gap target, using the known optimum; violation, gap and
    zeros are from the $(first(GAPS))% run. DC3 + correction also works on y directly.
    A timing cell needs every instance within the gap target and violation ≤ $C_V;
    **bold** marks the lowest mean time among those. — means results are missing.

    |  | $header |
    |---|$(repeat("---|", length(columns)))
    $(join(rows, "\n"))
    """
    write(joinpath(OUT, "qp_l1_table.md"), md)
    println(md)
end

if abspath(PROGRAM_FILE) == @__FILE__
    X = instances()
    files = ["OSQP.npz"; [slme_file(tag, gap) for (tag, _) in MODELS for gap in GAPS]]
    "--render" in ARGS || (("--force" in ARGS || !all(f -> isfile(joinpath(OUT, f)), files)) && run_julia(X))
    render(X)
end
