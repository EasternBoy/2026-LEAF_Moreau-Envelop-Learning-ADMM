# Moreau-envelope training data for QP + L1 (problems/qp_l1), following experiments/qp/data_gen_admm.jl.
# Run from the repository root: julia --project=. experiments/qp_l1/data_gen_admm.jl [--rho=10] [--lambda=1] [--gap=Inf] [--keep=1]
#   --rho, --lambda: ρ of the envelope/ADMM and the L1 weight λ (default QP_L1_RHO, QP_L1_LAMBDA)
#   --gap: also stop only once the objective gap to OSQP (tol 1e-8) is ≤ gap % (default Inf: no gap test)
#   --keep: fraction of each instance's distinct iterates kept, drawn at random (default 1: all)
# The prox has no closed form (dense Q): each one is solved by FISTA (problems/qp_l1/problem.jl).
# Collects 40000/10000 vector samples from separate train/test instance streams: every distinct
# iterate of each converged exact sMEL-ADMM run is a sample, and instances are drawn until the
# total is reached (the last instance contributes a random subset of its iterates).
# Exact sMEL-ADMM is problems/qp/lme_admm.jl with the exact prox in place of the ICNN, run with
# its stopping test at the benchmark settings of experiments/qp_l1/table.jl, so the samples
# cover the whole trajectory the learned solver follows.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Printf, Random, LinearAlgebra, SparseArrays, NPZ, Distributions
using LMEADMM

function gen_arg(name, default)
    i = findfirst(a -> startswith(a, "--$name="), ARGS)
    return i === nothing ? default : parse(Float64, split(ARGS[i], "="; limit = 2)[2])
end

const FloatType = Float64
include(joinpath(@__DIR__, "..", "..", "problems", "qp_l1", "problem.jl"))
include(joinpath(@__DIR__, "..", "..", "problems", "qp_l1", "setup.jl"))
include(joinpath(@__DIR__, "..", "..", "problems", "qp_l1", "jump_solver.jl"))

const REPO         = abspath(joinpath(@__DIR__, "..", ".."))
const OUT          = joinpath(REPO, "data", "qp_l1", "training")
const N_SIZE       = 100 # n; 50 equalities and 50 inequalities in the fixed family
const M_INEQ       = 50
const SEED         = 1
const N_TRAIN      = 40000
const N_TEST       = 10000
const SLME_TOL     = 1e-2   # sMEL-ADMM consensus tolerance, as in experiments/qp_l1/table.jl
const FEAS_TOL     = 1e-6   # sMEL-ADMM feasibility tolerance, as in experiments/qp_l1/table.jl
const MAX_ITER     = 5000
const RHO          = gen_arg("rho", QP_L1_RHO)
const LAMBDA       = gen_arg("lambda", QP_L1_LAMBDA)
const KEEP_FRAC    = gen_arg("keep", 1.0)
const GAP_STOP     = gen_arg("gap", Inf)
0 < KEEP_FRAC <= 1 || error("--keep must be in (0, 1]")
const NEQ          = 50

function instance(rng, n, m)
    return data_opt(; n, neq = NEQ, nineq = m, x = rand(rng, Uniform(-1, 1), NEQ), rho = RHO, λ = LAMBDA)
end

"sMEL-ADMM with the exact prox and the stopping test of sLME_ADMM; returns inputs (n × K), their envelope values and gradients, the final decision vector, and convergence diagnostics."
function admm_inputs(data::data_opt; max_iter = MAX_ITER, tol = SLME_TOL, feas_tol = FEAS_TOL,
                     J_ref = NaN, gap_stop = GAP_STOP)
    n, m, ρ = data.n, data.nineq, data.rho
    scale = var_scale(data)
    z = zeros(n + m); w = copy(z); v = copy(z); α = copy(z); β = copy(z)
    # [y; s/D], with D_i = ||G_i||₂: G*y + s = h, s >= 0.
    slack_scale = vec(sqrt.(sum(abs2, data.G; dims = 2)))
    M = [sparse(data.A) spzeros(data.neq, m); sparse(data.G ./ slack_scale) sparse(I, m, m)]
    proj = AffineProjection(kkt_matrix(M), scale .* [data.x; data.h ./ slack_scale])
    Q = Vector{Vector{FloatType}}(); E = FloatType[]; Gr = Vector{FloatType}[]
    primal_residual = dual_residual = Inf
    for i in 1:max_iter
        q = v[1:n] + β[1:n]
        x, env = prox_env(q; ρ = ρ, λ = data.λ)
        push!(Q, q); push!(E, env); push!(Gr, ρ*(q - x))
        z[1:n] .= x
        z[n+1:end] .= v[n+1:end] + β[n+1:end]
        v_previous = copy(v)
        w_previous = copy(w)
        v .= proj((z - β + w + α)/2)
        w[1:n] .= v[1:n] - α[1:n]
        w[n+1:end] .= max.(v[n+1:end] - α[n+1:end], 0.0)
        α .+= w - v
        β .+= v - z
        primal_residual = max(maximum(abs, w - v), maximum(abs, v - z))
        dual_residual = ρ * max(maximum(abs, v - v_previous), maximum(abs, w - w_previous))   # diagnostic only
        # sLME_ADMM's test: residual < tol and the decision vector feasible to feas_tol.
        y = v[1:n] / scale
        gap_ok = !isfinite(gap_stop) || 100abs(get_objective(data, y) - J_ref)/max(abs(J_ref), eps(FloatType)) <= gap_stop
        if primal_residual < tol && qp_solution_feasible(data, y, feas_tol) && gap_ok
            eq_viol = maximum(abs, data.A*y - data.x)
            ineq_viol = max(maximum(data.G*y - data.h), 0.0)
            return reduce(hcat, Q), E, reduce(hcat, Gr), y, (primal_residual = primal_residual,
                dual_residual = dual_residual, eq_viol = eq_viol, ineq_viol = ineq_viol)
        end
    end
    error("Exact sMEL-ADMM did not converge in $max_iter iterations: primal=$primal_residual, dual=$dual_residual; samples were not accepted")
end

function collect_split(split, seed, total)
    rng = Xoshiro(seed)
    X = Vector{FloatType}[]; E = FloatType[]; G = Vector{FloatType}[]
    meta = Dict(k => Int[] for k in ("n", "m", "neq", "instance", "iteration", "n_iter"))
    checks = Dict(k => FloatType[] for k in ("primal_residual", "dual_residual", "eq_viol", "ineq_viol", "opt_gap_percent"))
    k = 0
    while length(X) < total
        k += 1
        n, m = N_SIZE, M_INEQ
        data = instance(rng, n, m)
        # The gap stopping test (--gap) and a diagnostic (labels come from the exact prox): NaN when OSQP stops short.
        J_opt = try last(JuMP_solver("osqp", data, 1e-8)) catch; NaN end
        isfinite(GAP_STOP) && !isfinite(J_opt) && continue   # no reference for the gap test
        Q, env, grad, y, status = admm_inputs(data; J_ref = J_opt)
        gap = 100abs(get_objective(data, y) - J_opt)/max(abs(J_opt), eps(FloatType))
        K = size(Q, 2)
        pool = unique(t -> Q[:, t], collect(1:K))
        n_distinct = length(pool)
        if KEEP_FRAC < 1   # a random KEEP_FRAC share of this instance's distinct iterates
            pool = sort(pool[randperm(rng, length(pool))[1:max(1, round(Int, KEEP_FRAC * length(pool)))]])
        end
        keep = length(X) + length(pool) <= total ? pool : sort(pool[randperm(rng, length(pool))[1:total - length(X)]])
        for idx in keep
            for key in ("primal_residual", "dual_residual", "eq_viol", "ineq_viol")
                push!(checks[key], getproperty(status, Symbol(key)))
            end
            push!(checks["opt_gap_percent"], gap)
            push!(X, Q[:, idx]); push!(E, env[idx]); push!(G, grad[:, idx])
            for (key, val) in zip(("n", "m", "neq", "instance", "iteration", "n_iter"), (n, m, NEQ, k, idx, K))
                push!(meta[key], val)
            end
        end
        @printf("%-5s instance %4d: ADMM %4d it., %4d distinct q, kept %4d, total %6d | primal=%.1e dual=%.1e eq=%.1e OSQP gap=%.3g%%\n",
                split, k, K, n_distinct, length(keep), length(X), status.primal_residual, status.dual_residual, status.eq_viol, gap)
    end
    path = joinpath(OUT, "qp_l1-lambda=$(LAMBDA)-rho=$(RHO)" * (isfinite(GAP_STOP) ? "-gap=$(GAP_STOP)" : "") * (KEEP_FRAC < 1 ? "-keep=$(KEEP_FRAC)" : "") * "-$(split).npz")
    npzwrite(path, merge(Dict("input" => reduce(hcat, X), "grad" => reduce(hcat, G), "enve" => E, "rho" => RHO,
                              "seed" => seed, "full_trajectory" => KEEP_FRAC == 1, "instances" => k,
                              "Q" => qp_data["Q"], "p" => qp_data["p"], "A" => qp_data["A"],
                              "G" => qp_data["G"], "h" => qp_data["h"], "lambda" => LAMBDA, "keep_frac" => KEEP_FRAC, "gap_stop_pct" => GAP_STOP, "tol" => SLME_TOL, "feas_tol" => FEAS_TOL, "max_iter" => MAX_ITER,
                              "normalized_slacks" => true, "scaled_duals" => true), meta, checks))
    @printf("wrote %s: %d samples from %d instances, q in [%.3f, %.3f]\n", path, length(X), k, minimum(minimum, X), maximum(maximum, X))
end

mkpath(OUT)
collect_split("train", SEED, N_TRAIN)
collect_split("test",  SEED + 1, N_TEST)
