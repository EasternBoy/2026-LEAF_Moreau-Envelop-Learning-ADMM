# Moreau-envelope QP training data, following entr_max/data_gen_admm.jl.
# Run from the repository root: julia --project=. experiments/qp/data_gen_admm.jl
# Collects 8000/2000 vector samples from separate train/test instance streams.
# Retains FIRST_FRAC of each converged exact split-ADMM trajectory, ten vectors per instance.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Printf, Random, LinearAlgebra, SparseArrays, NPZ, Distributions
using LMEADMM

const FloatType = Float64
include(joinpath(@__DIR__, "..", "..", "problems", "qp", "problem.jl"))
include(joinpath(@__DIR__, "..", "..", "problems", "qp", "setup.jl"))
include(joinpath(@__DIR__, "..", "..", "problems", "qp", "jump_solver.jl"))

const REPO         = abspath(joinpath(@__DIR__, "..", ".."))
const OUT          = joinpath(REPO, "data", "qp", "training")
const SIZES        = [(100, 50)] # (n, inequality count); 50 equalities in the fixed family
const SHARES       = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [1]
const TAG          = allequal(SHARES) ? "" : "-" * join(SHARES)
const SEED         = 1
const N_TRAIN      = 8000
const N_TEST       = 2000
const PER_INSTANCE = 10
const FIRST_FRAC   = 1.
const SLME_TOL     = 1e-3
const MAX_ITER     = 1000
const RHO          = 1.0
const NEQ          = 50

"prox and Moreau envelope of the fixed diagonal QP objective."
function prox_env(q::Vector{FloatType}; ρ = RHO)
    x = (ρ .* q .- qp_data["p"]) ./ (diag(qp_data["Q"]) .+ ρ)
    env = 0.5*dot(x, qp_data["Q"]*x) + dot(qp_data["p"], x) + ρ/2*sum(abs2, x - q)
    return x, env
end

function instance(rng, n, m)
    return data_opt(; n, neq = NEQ, nineq = m, x = rand(rng, Uniform(-1, 1), NEQ), rho = RHO)
end

"sLME-ADMM with the exact prox; returns inputs (n × K), the final decision vector, and convergence diagnostics."
function admm_inputs(data::data_opt; max_iter = MAX_ITER, tol = SLME_TOL)
    n, m, ρ = data.n, data.nineq, data.rho
    scale = var_scale(data)
    z = zeros(n + m); w = copy(z); v = copy(z); α = copy(z); β = copy(z)
    # [y; s/D], with D_i = ||G_i||₂: G*y + s = h, s >= 0.
    slack_scale = vec(sqrt.(sum(abs2, data.G; dims = 2)))
    M = [sparse(data.A) spzeros(data.neq, m); sparse(data.G ./ slack_scale) sparse(I, m, m)]
    proj = AffineProjection(kkt_matrix(M), scale .* [data.x; data.h ./ slack_scale])
    Q = Vector{Vector{FloatType}}()
    primal_residual = dual_residual = Inf
    for i in 1:max_iter
        q = v[1:n] + β[1:n]
        push!(Q, q)
        z[1:n] .= first(prox_env(q; ρ = ρ))
        z[n+1:end] .= v[n+1:end] + β[n+1:end]
        v_previous = copy(v)
        w_previous = copy(w)
        v .= proj((z - β + w + α)/2)
        w[1:n] .= v[1:n] - α[1:n]
        w[n+1:end] .= max.(v[n+1:end] - α[n+1:end], 0.0)
        α .+= w - v
        β .+= v - z
        primal_residual = max(maximum(abs, w - v), maximum(abs, v - z))
        # Track both moving blocks in this z -> v -> w update order.
        dual_residual = ρ * max(maximum(abs, v - v_previous), maximum(abs, w - w_previous))
        if max(primal_residual, dual_residual) < tol*scale
            y = v[1:n] / scale
            eq_viol = maximum(abs, data.A*y - data.x)
            ineq_viol = max(maximum(data.G*y - data.h), 0.0)
            max(eq_viol, ineq_viol) < tol || continue
            return reduce(hcat, Q), y, (primal_residual = primal_residual,
                dual_residual = dual_residual, eq_viol = eq_viol, ineq_viol = ineq_viol)
        end
    end
    error("Exact split ADMM did not converge in $max_iter iterations: primal=$primal_residual, dual=$dual_residual; samples were not accepted")
end

"Instances per size for `total` samples, and the round-robin order in which they are drawn."
function size_order(total)
    counts = total .* SHARES .÷ (sum(SHARES) * PER_INSTANCE)
    counts .* (sum(SHARES) * PER_INSTANCE) == total .* SHARES || error("$total samples do not split into $SHARES")
    order = Int[]
    while length(order) < sum(counts)
        for j in eachindex(counts)
            count(==(j), order) < counts[j] && push!(order, j)
        end
    end
    return order
end

function collect_split(split, seed, total)
    rng = Xoshiro(seed)
    X = Vector{FloatType}[]; E = FloatType[]; G = Vector{FloatType}[]
    meta = Dict(k => Int[] for k in ("n", "m", "neq", "instance", "iteration", "n_iter"))
    checks = Dict(k => FloatType[] for k in ("primal_residual", "dual_residual", "eq_viol", "ineq_viol", "opt_gap_percent"))
    for (k, j) in enumerate(size_order(total))
        n, m = SIZES[j]
        data = instance(rng, n, m)
        Q, y, status = admm_inputs(data)
        _, _, J_opt = JuMP_solver("osqp", data, 1e-9)
        gap = 100abs(get_objective(data, y) - J_opt)/max(abs(J_opt), eps(FloatType))
        @printf("  converged: primal=%.3e, dual=%.3e, eq=%.3e, ineq=%.3e, OSQP gap=%.6g%%\n",
                status.primal_residual, status.dual_residual, status.eq_viol, status.ineq_viol, gap)
        K = size(Q, 2)
        K_keep = ceil(Int, FIRST_FRAC*K)
        pool = unique(t -> Q[:, t], collect(1:K_keep))
        length(pool) >= PER_INSTANCE || error("instance $k: only $(length(pool)) distinct inputs")
        for idx in pool[randperm(rng, length(pool))[1:PER_INSTANCE]]
            for key in ("primal_residual", "dual_residual", "eq_viol", "ineq_viol")
                push!(checks[key], getproperty(status, Symbol(key)))
            end
            push!(checks["opt_gap_percent"], gap)
            q = Q[:, idx]
            x, env = prox_env(q)
            push!(X, q); push!(E, env); push!(G, RHO*(q - x))
            for (key, val) in zip(("n", "m", "neq", "instance", "iteration", "n_iter"), (n, m, NEQ, k, idx, K))
                push!(meta[key], val)
            end
        end
        @printf("%-5s instance %3d (n=%4d, m=%3d): ADMM %4d it., kept first %3d, %6d distinct q\n",
                split, k, n, m, K, K_keep, length(pool))
    end
    path = joinpath(OUT, "qp$(TAG)-rho=$(RHO)-$(split).npz")
    npzwrite(path, merge(Dict("input" => reduce(hcat, X), "grad" => reduce(hcat, G), "enve" => E, "rho" => RHO,
                              "seed" => seed, "first_frac" => FIRST_FRAC,
                              "Q" => qp_data["Q"], "p" => qp_data["p"], "A" => qp_data["A"],
                              "G" => qp_data["G"], "h" => qp_data["h"], "tol" => SLME_TOL, "max_iter" => MAX_ITER,
                              "normalized_slacks" => true, "scaled_duals" => true), meta, checks))
    @printf("wrote %s: %d samples, q in [%.3f, %.3f]\n", path, length(X), minimum(minimum, X), maximum(maximum, X))
end

mkpath(OUT)
collect_split("train", SEED, N_TRAIN)
collect_split("test",  SEED + 1, N_TEST)
