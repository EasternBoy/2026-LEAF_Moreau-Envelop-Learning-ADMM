# Solving-time / optimality-gap table for the maximum-entropy cone program:
# the IPOPT and sLME-ADMM columns.  The DC3 column is filled by CP_table.py.
#
#   julia --project=. script/entropy_max/CP_table.jl            # use stored data, run what is missing
#   julia --project=. script/entropy_max/CP_table.jl --force    # recompute everything
#
# Internal mode (one (n, m) per process, because `n` is a `const` in
# examples/cone_programming/maxEntropy.jl):
#
#   julia --project=. --threads=auto script/entropy_max/CP_table.jl worker n m [--instances-only] [--force]
#
# Data and the rendered table live in data/cone_result (see README.md there).

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Base.Threads
using Printf, Random, SparseArrays, JSON3, LinearAlgebra, StaticArrays, NPZ, JuMP, NNlib
using Distributions, Statistics
import ParametricOptInterface as POI
import MathOptInterface       as MOI

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const OUT      = joinpath(REPO, "data", "cone_result")
const INST_DIR = joinpath(OUT, "instances")
cd(REPO)   # utils.jl loads the ICNN with a path relative to the repository root

const SIZES    = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
const GAP_ROWS = [(100, 10), (1000, 100)]
const GOPTS    = [(1.0, "1"), (0.1, "0.1")]      # g_opt in %, and its spelling in file names
const METHODS  = ["IPOPT", "sLME-ADMM"]
const N_SAMPLES = 1000
const SEED      = 20260923
const FEAS_TOL  = 1e-5       # IPOPT early stop: violation below FEAS_TOL (inf_pr < FEAS_TOL*scale)
const REPORT_TOL = 1e-4      # threshold for the feasible rate printed in the log
const SLME_TOL  = 1e-5       # sLME-ADMM residual tolerance on w: tol = SLME_TOL*scale

inst_file(n, m)           = joinpath(INST_DIR, "instances-n=$(n)-m=$(m).npz")
gt_file(n, m)             = joinpath(OUT, "ground_truth-n=$(n)-m=$(m).npz")
res_file(meth, g, n, m)   = joinpath(OUT, "$(meth)-gopt=$(g)-n=$(n)-m=$(m).npz")
dc3_file(n, m)            = joinpath(OUT, "DC3-n=$(n)-m=$(m).npz")

const IS_WORKER = length(ARGS) >= 3 && ARGS[1] == "worker"
const FORCE     = "--force" in ARGS

# ---------------------------------------------------------------------------
# worker: globals expected by examples/cone_programming/*.jl
const FloatType = Float64
const n::Int    = IS_WORKER ? parse(Int, ARGS[2]) : 0
const m::Int    = IS_WORKER ? parse(Int, ARGS[3]) : 0
# benchmarkOG.jl uses max(div(n, nthreads())+1, 50); clamped to n so that
# utils.jl::mini_batch stays in bounds for small n.
const s_mb::Int = IS_WORKER ? min(max(div(n, nthreads()) + 1, 50), n) : 1
max_opt_gap::FloatType = 1.0                     # read by the callbacks in preprocess.jl
GUROBI_ENV = nothing                             # Gurobi is not needed (no license)

IS_WORKER && include(joinpath(REPO, "examples", "cone_programming", "maxEntropy.jl"))
IS_WORKER && include(joinpath(REPO, "examples", "cone_programming", "preprocess.jl"))
IS_WORKER && include(joinpath(REPO, "examples", "cone_programming", "JuMPsolver.jl"))
IS_WORKER && include(joinpath(REPO, "examples", "cone_programming", "LME-ADMM.jl"))
IS_WORKER && (ipopt_feas_tol = FEAS_TOL)

entropy(w) = all(isfinite, w) && all(>=(0), w) ? sum(x -> x == 0 ? 0.0 : x * log(x), w) : NaN
max_violation(A, b, w) = max(maximum(A * w .- b), maximum(-w), abs(sum(w) - 1), 0.0)

function instances()
    path = inst_file(n, m)
    if isfile(path) && !FORCE
        d = npzread(path)
        return d["A"], d["b"]
    end
    Random.seed!(SEED)
    A = zeros(N_SAMPLES, m, n); b = zeros(N_SAMPLES, m)
    for k in 1:N_SAMPLES
        para = data_opt(n, m)                    # the repository's instance distribution
        A[k, :, :] = para.A; b[k, :] = para.b
    end
    mkpath(INST_DIR)
    npzwrite(path, Dict("A" => A, "b" => b))
    return A, b
end

instance(A, b, k) = data_opt(n, m, Matrix{FloatType}(A[k, :, :]), Vector{FloatType}(b[k, :]), 1.0, x -> x * log(x))

function ground_truth(A, b)
    path = gt_file(n, m)
    if isfile(path) && !FORCE
        return npzread(path)["J_opt"]
    end
    J = zeros(N_SAMPLES); t = zeros(N_SAMPLES)
    JuMP_solver("Ipopt", instance(A, b, 1), 1e-8)            # compile
    for k in 1:N_SAMPLES
        _, t[k], J[k] = JuMP_solver("Ipopt", instance(A, b, k), 1e-8)
        k % 5 == 0 && GC.gc()
    end
    npzwrite(path, Dict("J_opt" => J, "time_ms" => 1e3 .* t, "seed" => SEED))
    @printf("n=%d m=%d: ground truth (Ipopt, tol 1e-8) done\n", n, m)
    return J
end

function solve_one(meth, para, mgrad)
    if meth == "IPOPT"
        w, t, _ = JuMP_solver("Ipopt", para, 1e-2, callback_struct())   # stopped by Ipopt_callback_BM
        return w, t, 0
    end
    it = Ref(0)
    sol, t, _ = sLME_ADMM(para, mgrad, (args...) -> (it[] = args[6]; sLME_ADMM_callback(args...));
                          tol = SLME_TOL * scale)
    return sol[1:n], t, it[]
end

function run_method(meth, gopt, gname, A, b, Jopt, mgrad)
    path = res_file(meth, gname, n, m)
    (isfile(path) && !FORCE && get(npzread(path), "metrics_version", 0) == 2) && return
    global max_opt_gap = gopt
    global J_opt = Jopt[1]
    solve_one(meth, instance(A, b, 1), mgrad)                # compile, not recorded
    t_ms = zeros(N_SAMPLES); gap = zeros(N_SAMPLES); viol = zeros(N_SAMPLES); iters = zeros(Int, N_SAMPLES); domain_valid = falses(N_SAMPLES)
    for k in 1:N_SAMPLES
        global J_opt = Jopt[k]
        para = instance(A, b, k)
        w, t, iters[k] = solve_one(meth, para, mgrad)
        domain_valid[k] = all(isfinite, w) && all(>=(0), w)
        t_ms[k] = 1e3t
        gap[k]  = 100abs(entropy(w) - Jopt[k]) / abs(Jopt[k])
        viol[k] = max_violation(para.A, para.b, w)
        k % 5 == 0 && GC.gc()
    end
    npzwrite(path, Dict("time_ms" => t_ms, "gap_pct" => gap, "max_viol" => viol, "iterations" => iters, "domain_valid" => domain_valid, "feasible" => domain_valid .& (viol .<= REPORT_TOL), "metrics_version" => 2, "oracle_assisted" => true))
    @printf("n=%d m=%d %-9s g_opt=%s%%: time %.3f (%.3f) ms, gap %.3g (%.3g) %%, feasible %.3f\n",
            n, m, meth, gname, mean(t_ms), maximum(t_ms), mean(gap), maximum(gap), mean(domain_valid .& (viol .<= REPORT_TOL)))
end

function worker()
    A, b = instances()
    Jopt = ground_truth(A, b)
    "--instances-only" in ARGS && return
    mgrad = gradient_struct(model, s_mb, 1)
    for meth in METHODS, (gopt, gname) in GOPTS
        run_method(meth, gopt, gname, A, b, Jopt, mgrad)
    end
end

# ---------------------------------------------------------------------------
# driver + table
function run_missing()
    for (nn, mm) in SIZES
        need = [gt_file(nn, mm); [res_file(me, g, nn, mm) for me in METHODS for (_, g) in GOPTS]]
        result_files = need[2:end]
        current = all(f -> isfile(f) && get(npzread(f), "metrics_version", 0) == 2, result_files)
        (FORCE || !all(isfile, need) || !current) || continue
        cmd = `$(Base.julia_cmd()) --project=$REPO --threads=auto $(@__FILE__) worker $nn $mm`
        FORCE && (cmd = `$cmd --force`)
        run(cmd)
    end
end

fmt_time(x) = @sprintf("%.2f (%.2f)", mean(x), maximum(x))
fmt_gap(x)  = all(isfinite, x) ? @sprintf("%.3g (%.3g)", mean(x), maximum(x)) : "undefined (outside objective domain)"
fmt_viol(x) = @sprintf("%.1e (%.1e)", mean(x), maximum(x))

function dc3_cell(nn, mm, gopt)
    isfile(dc3_file(nn, mm)) || return "—", Inf
    d = npzread(dc3_file(nn, mm))
    get(d, "metrics_version", 0) == 2 || return "rerun required", Inf
    ok = all(d["feasible"] .> 0.5) && maximum(d["gap_pct"]) <= gopt
    return ok ? (fmt_time(d["time_ms"]), mean(d["time_ms"])) : ("unable to achieve", Inf)
end

function time_cell(meth, gname, nn, mm)
    isfile(res_file(meth, gname, nn, mm)) || return "—", Inf
    d = npzread(res_file(meth, gname, nn, mm))
    get(d, "metrics_version", 0) == 2 || return "rerun required", Inf
    (all(d["feasible"] .> 0.5) && all(d["gap_pct"] .<= parse(Float64, gname))) || return "unable to achieve", Inf
    t = d["time_ms"]
    return fmt_time(t), mean(t)
end

function iter_suffix(gname, nn, mm)
    isfile(res_file("sLME-ADMM", gname, nn, mm)) || return ""
    return @sprintf(" [%.1f it.]", mean(npzread(res_file("sLME-ADMM", gname, nn, mm))["iterations"]))
end

function metric_cell(path, key, formatter)
    d = npzread(path)
    return get(d, "metrics_version", 0) == 2 ? formatter(d[key]) : "rerun required"
end

function render()
    rows = String[]
    for (gopt, gname) in GOPTS, (nn, mm) in SIZES
        cells = [time_cell("IPOPT", gname, nn, mm), time_cell("sLME-ADMM", gname, nn, mm), dc3_cell(nn, mm, gopt)]
        best = argmin(last.(cells))
        txt = [isfinite(c[2]) && i == best ? "**$(c[1])**" : c[1] for (i, c) in enumerate(cells)]
        txt[2] *= iter_suffix(gname, nn, mm)
        push!(rows, "| solving time (g_opt ≤ $(gname)%) | $nn | $mm | $(join(txt, " | ")) |")
    end
    for (nn, mm) in GAP_ROWS
        ip = isfile(gt_file(nn, mm)) ? "0" : "—"
        sl = isfile(res_file("sLME-ADMM", "0.1", nn, mm)) ? metric_cell(res_file("sLME-ADMM", "0.1", nn, mm), "gap_pct", fmt_gap) : "—"
        dc = "—"
        if isfile(dc3_file(nn, mm))
            d = npzread(dc3_file(nn, mm))
            dc = get(d, "metrics_version", 0) == 2 ? fmt_gap(d["gap_pct"]) : "rerun required"
            f = mean(d["feasible"] .> 0.5)
            f < 1 && (dc *= @sprintf(", feasible %.0f%%", 100f))
        end
        push!(rows, "| Opt. gap (%) | $nn | $mm | $ip | $sl | $dc |")
    end
    for (_, gname) in GOPTS, (nn, mm) in SIZES
        cells = [isfile(f) ? metric_cell(f, "max_viol", fmt_viol) : "—"
                 for f in (res_file("IPOPT", gname, nn, mm), res_file("sLME-ADMM", gname, nn, mm), dc3_file(nn, mm))]
        push!(rows, "| Constr. viol. (g_opt ≤ $(gname)%) | $nn | $mm | $(join(cells, " | ")) |")
    end
    md = """
    # Maximum-entropy cone program: solving time and optimality gap

    Generated by `script/entropy_max/CP_table.jl` (IPOPT, sLME-ADMM) and
    `script/entropy_max/CP_table.py` (DC3) from the data in this folder — see
    [README.md](README.md) for what each number means.  Solving time in ms and
    optimality gap in %, as mean (max) over $(N_SAMPLES) instances per row; **bold** is
    the lowest mean time in the row; `[k it.]` is sLME-ADMM's mean number of
    iterations; — means the data has not been produced yet.  Constr. viol. is the
    largest violation `max(max(A w − b), max(−w), |1ᵀw − 1|)` of each returned point
    (IPOPT and sLME-ADMM from the run with that g_opt; DC3 has a single run).
    IPOPT and sLME-ADMM use oracle-assisted stopping against the known optimum;
    reference-solve cost is excluded. Entropy gaps require w >= 0 exactly.

    |  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 mean (max) |
    |---|---|---|---|---|---|
    $(join(rows, "\n"))
    """
    write(joinpath(OUT, "CP_table.md"), md)
    println(md)
end

IS_WORKER ? worker() : (run_missing(); render())
