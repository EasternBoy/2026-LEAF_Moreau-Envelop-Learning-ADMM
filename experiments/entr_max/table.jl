using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Base.Threads
using Printf, Random, SparseArrays, JSON3, LinearAlgebra, StaticArrays, NPZ, JuMP, NNlib
using Distributions, Statistics
import ParametricOptInterface as POI
import MathOptInterface       as MOI

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const OUT      = joinpath(REPO, "results", "entr_max", "table")
const INST_DIR = joinpath(OUT, "instances")
cd(REPO)   # utils.jl loads the ICNN with a path relative to the repository root

const SIZES    = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
const GAP_ROWS = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
const GOPTS    = [(1.0, "1"), (0.1, "0.1")]      # g_opt in %, and its spelling in file names
const METHODS  = ["IPOPT", "sLME-ADMM"]
const N_SAMPLES = 1000
const SEED      = 20260923
const IPOPT_FEAS_TOL = 1e-5  # IPOPT early stop: violation below IPOPT_FEAS_TOL (inf_pr < IPOPT_FEAS_TOL*scale)
const SLME_TOL  = 1e-5       # sLME-ADMM residual tolerance on w: tol = SLME_TOL*scale

inst_file(n, m)           = joinpath(INST_DIR, "instances-n=$(n)-m=$(m).npz")
gt_file(n, m)             = joinpath(OUT, "ground_truth-n=$(n)-m=$(m).npz")
# --model=<path>: the ICNN sLME-ADMM uses (default: the one in problems/entr_max/utils.jl);
# --tag=<name>: its sLME-ADMM results and table go to OUT/<name>, the other methods are read from OUT.
argval(key) = (i = findfirst(a -> startswith(a, "--$key="), ARGS); i === nothing ? nothing : String(split(ARGS[i], "="; limit = 2)[2]))
const MODEL_ARG = argval("model")
const TAG       = something(argval("tag"), "")
const OUT_TAG   = isempty(TAG) ? OUT : joinpath(OUT, TAG)
const PASS_ARGS = filter(a -> startswith(a, "--model=") || startswith(a, "--tag="), ARGS)
MODEL_ARG === nothing || (ICNN_MODEL = MODEL_ARG)   # read by problems/entr_max/utils.jl

res_file(meth, g, n, m)   = joinpath(meth == "sLME-ADMM" ? OUT_TAG : OUT, "$(meth)-gopt=$(g)-n=$(n)-m=$(m).npz")
dc3_file(n, m)            = joinpath(OUT, "DC3-n=$(n)-m=$(m).npz")

using LMEADMM   # src/metrics.jl: gap, violation, feasibility for every method
is_current(path) = isfile(path) && get(npzread(path), "metrics_version", 0) == METRICS_VERSION

const IS_WORKER = length(ARGS) >= 3 && ARGS[1] == "worker"
const FORCE     = "--force" in ARGS

# ---------------------------------------------------------------------------
# worker: globals expected by problems/entr_max/*.jl
const FloatType = Float64
const n::Int    = IS_WORKER ? parse(Int, ARGS[2]) : 0
const m::Int    = IS_WORKER ? parse(Int, ARGS[3]) : 0
# benchmark.jl uses max(div(n, nthreads())+1, 50); clamped to n so that
# utils.jl::mini_batch stays in bounds for small n.
const s_mb::Int = IS_WORKER ? min(max(div(n, nthreads()) + 1, 50), n) : 1
max_opt_gap::FloatType = 1.0                     # read by the callbacks in problems/entr_max/setup.jl
GUROBI_ENV = nothing                             # Gurobi is not needed (no license)

IS_WORKER && include(joinpath(REPO, "problems", "entr_max", "problem.jl"))
IS_WORKER && include(joinpath(REPO, "problems", "entr_max", "setup.jl"))
IS_WORKER && include(joinpath(REPO, "problems", "entr_max", "jump_solver.jl"))
IS_WORKER && include(joinpath(REPO, "problems", "entr_max", "lme_admm.jl"))
IS_WORKER && (ipopt_feas_tol = IPOPT_FEAS_TOL)

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

function save_result(meth, gname, A, b, Jopt, W, t_ms, iters)
    path = res_file(meth, gname, n, m)
    s = score_entr_max(A, b, W, Jopt)
    mkpath(dirname(path))
    npzwrite(path, Dict("W" => W, "time_ms" => t_ms, "iterations" => iters, "gap_pct" => s.gap_pct, "max_viol" => s.max_viol,
                        "feasible" => s.feasible, "metrics_version" => METRICS_VERSION, "oracle_assisted" => true))
    @printf("n=%d m=%d %-9s g_opt=%s%%: time %.3f (%.3f) ms, gap %.3g (%.3g) %%, feasible %.3f\n",
            n, m, meth, gname, mean(t_ms), maximum(t_ms), mean(s.gap_pct), maximum(s.gap_pct), mean(s.feasible))
end

function run_ipopt(gopt, gname, A, b, Jopt)
    is_current(res_file("IPOPT", gname, n, m)) && !FORCE && return
    global max_opt_gap = gopt
    global J_opt = Jopt[1]
    JuMP_solver("Ipopt", instance(A, b, 1), 1e-2, callback_struct())   # compile, not recorded
    t_ms = zeros(N_SAMPLES); W = zeros(N_SAMPLES, n)
    for k in 1:N_SAMPLES
        global J_opt = Jopt[k]
        w, t, _ = JuMP_solver("Ipopt", instance(A, b, k), 1e-2, callback_struct())   # stopped by Ipopt_callback_BM
        W[k, :] = w
        t_ms[k] = 1e3t
        k % 5 == 0 && GC.gc()
    end
    save_result("IPOPT", gname, A, b, Jopt, W, t_ms, zeros(Int, N_SAMPLES))
end

# sLME-ADMM runs once per instance, to the strictest g_opt.  Each g_opt gets the time, iteration
# count and solution of the first iteration where the gap is below it and the residual below its
# tolerance, so a looser g_opt is never reported slower than a stricter one.
function slme_one!(T, W, IT, k, para, mgrad)
    tol, scale = SLME_TOL * var_scale(para), var_scale(para)
    got = falses(length(GOPTS)); last_it = Ref(0)
    cb = (z, w, α, v, β, i, J, elapsed) -> begin
        last_it[] = i
        gap = 100abs(J - J_opt)/abs(J_opt)
        res = max(maximum(abs(x - y) for (x, y) in zip(w, v)), maximum(abs(x - y) for (x, y) in zip(v, z)))
        for (j, (g, _)) in enumerate(GOPTS)
            if !got[j] && gap < g && res < tol
                got[j] = true; T[k, j] = 1e3elapsed; IT[k, j] = i
                @views W[k, :, j] .= v[1:n] ./ scale
            end
        end
        return sLME_ADMM_callback(z, w, α, v, β, i, J)
    end
    sol, t, _ = sLME_ADMM(para, mgrad, cb; tol = tol)
    for j in findall(!, got)                                 # max_iter reached before this g_opt
        T[k, j] = 1e3t; IT[k, j] = last_it[]; W[k, :, j] .= sol[1:n]
    end
end

function run_slme(A, b, Jopt, mgrad)
    all(is_current(res_file("sLME-ADMM", gname, n, m)) for (_, gname) in GOPTS) && !FORCE && return
    global max_opt_gap = minimum(first, GOPTS)
    G = length(GOPTS)
    T = zeros(N_SAMPLES, G); W = zeros(N_SAMPLES, n, G); IT = zeros(Int, N_SAMPLES, G)
    global J_opt = Jopt[1]
    slme_one!(T, W, IT, 1, instance(A, b, 1), mgrad)         # compile, overwritten below
    for k in 1:N_SAMPLES
        global J_opt = Jopt[k]
        slme_one!(T, W, IT, k, instance(A, b, k), mgrad)
        k % 5 == 0 && GC.gc()
    end
    for (j, (_, gname)) in enumerate(GOPTS)
        save_result("sLME-ADMM", gname, A, b, Jopt, W[:, :, j], T[:, j], IT[:, j])
    end
end

function worker()
    A, b = instances()
    Jopt = ground_truth(A, b)
    "--instances-only" in ARGS && return
    for (gopt, gname) in GOPTS
        run_ipopt(gopt, gname, A, b, Jopt)
    end
    run_slme(A, b, Jopt, gradient_struct(model, s_mb, 1))
end

# ---------------------------------------------------------------------------
# driver + table
function run_missing()
    for (nn, mm) in SIZES
        need = [gt_file(nn, mm); [res_file(me, g, nn, mm) for me in METHODS for (_, g) in GOPTS]]
        (FORCE || !isfile(need[1]) || !all(is_current, need[2:end])) || continue
        cmd = `$(Base.julia_cmd()) --project=$REPO --threads=auto $(@__FILE__) worker $nn $mm $PASS_ARGS`
        FORCE && (cmd = `$cmd --force`)
        run(cmd)
    end
end

fmt_time(x) = @sprintf("%.2f (%.2f)", mean(x), maximum(x))
fmt_gap(x)  = all(isfinite, x) ? @sprintf("%.3g (%.3g)", mean(x), maximum(x)) : "undefined (non-finite solution)"
fmt_viol(x) = @sprintf("%.1e (%.1e)", mean(x), maximum(x))

function dc3_cell(nn, mm, gopt)
    isfile(dc3_file(nn, mm)) || return "—", Inf
    d = npzread(dc3_file(nn, mm))
    is_current(dc3_file(nn, mm)) || return "rerun required", Inf
    ok = all(d["feasible"] .> 0.5) && maximum(d["gap_pct"]) <= gopt
    return ok ? (fmt_time(d["time_ms"]), mean(d["time_ms"])) : ("unable to achieve", Inf)
end

function time_cell(meth, gname, nn, mm)
    isfile(res_file(meth, gname, nn, mm)) || return "—", Inf
    d = npzread(res_file(meth, gname, nn, mm))
    is_current(res_file(meth, gname, nn, mm)) || return "rerun required", Inf
    (all(d["feasible"] .> 0.5) && all(d["gap_pct"] .<= parse(Float64, gname))) || return "unable to achieve", Inf
    t = d["time_ms"]
    return fmt_time(t), mean(t)
end

function iter_suffix(gname, nn, mm)
    isfile(res_file("sLME-ADMM", gname, nn, mm)) || return ""
    return @sprintf(" [%.1f it.]", mean(npzread(res_file("sLME-ADMM", gname, nn, mm))["iterations"]))
end

metric_cell(path, key, formatter) = is_current(path) ? formatter(npzread(path)[key]) : "rerun required"

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
            dc = is_current(dc3_file(nn, mm)) ? fmt_gap(d["gap_pct"]) : "rerun required"
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

    Generated by `experiments/entr_max/table.jl` (IPOPT, sLME-ADMM) and
    `experiments/entr_max/table.py` (DC3) from the data in this folder — see
    [README.md]($(isempty(TAG) ? "" : "../")README.md) for what each number means.$(MODEL_ARG === nothing ? "" :
    "  sLME-ADMM uses the ICNN `$(MODEL_ARG)`; IPOPT and DC3 are the runs in the parent folder.")  Solving time in ms and
    optimality gap in %, as mean (max) over $(N_SAMPLES) instances per row; **bold** is
    the lowest mean time in the row; `[k it.]` is sLME-ADMM's mean number of
    iterations; — means the data has not been produced yet.  Constr. viol. is the
    largest violation `max(max(A w − b), max(−w), |1ᵀw − 1|)` of each returned point
    (IPOPT and sLME-ADMM from the run with that g_opt; DC3 has a single run).
    IPOPT and sLME-ADMM use oracle-assisted stopping against the known optimum;
    reference-solve cost is excluded. Every method's entropy is scored at max(w, 0);
    negative entries count in Constr. viol., and a point is feasible when that is ≤ $(FEAS_TOL).
    All three methods are scored by `src/metrics.jl` from their saved solutions.

    |  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
    |---|---|---|---|---|---|
    $(join(rows, "\n"))
    """
    mkpath(OUT_TAG)
    write(joinpath(OUT_TAG, "entr_max_table.md"), md)
    println(md)
end

# row.jl includes this file for its helpers; only run when executed directly.
if abspath(PROGRAM_FILE) == @__FILE__
    IS_WORKER ? worker() : (run_missing(); render())
end
