# Moreau-envelope training data for entr_max, collected in the ADMM loop.
#
#   julia --project=. experiments/entr_max/data_gen_admm.jl            # equal shares of the sizes
#   julia --project=. experiments/entr_max/data_gen_admm.jl 1,1,4,4    # shares of (100,1) (100,10) (1000,10) (1000,100)
#
# Writes data/entr_max/training/maxEntropy-<tag>-rho=1.0-{train,test}.npz, <tag> = admm20 for equal
# shares and admm20-<shares> otherwise (admm20-1144): 8000 train and 2000 test samples, split
# between the sizes in proportion to the shares (input, enve, grad, rho, as the other data files,
# plus where each sample came from: n, m, instance, iteration, n_iter).
#
# * Split like DC3: train and test instances come from two separate RNG streams
#   (Xoshiro(SEED), Xoshiro(SEED + 1)), both different from the table's instances
#   (seed 20260923), so no instance is shared between train, test and the benchmark.
#   Instances are drawn as in data_opt(n, m), round-robin over the table's (n, m) sizes until each
#   size has its share.
# * The ADMM is the sLME-ADMM iteration of problems/entr_max/lme_admm.jl with the exact prox
#   in place of the ICNN, run until its residual is below SLME_TOL·scale (as in table.jl),
#   or MAX_ITER.  If it takes K iterations, only the prox inputs q of the first
#   ceil(FIRST_FRAC·K) iterations are kept; duplicates are dropped (the first iterate is q = 0)
#   and PER_INSTANCE samples are drawn from what is left.
# * Labels: the Moreau envelope of f(x) = x log(x/S0) with S0 = 2000 (= var_scale at n = 1000),
#   the function the current models learned (data_gen! in problems/entr_max/admm.jl).  sLME-ADMM
#   uses this one model at every n: the other scales differ from it by a term linear in x, which
#   is constant on 1ᵀx = scale.  The prox is solved by Newton's method in log x (exact to 1e-12);
#   the ADMM runs with the same prox, so the iterates are the ones sLME-ADMM approximates.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Printf, Random, LinearAlgebra, SparseArrays, NPZ, Distributions
using LMEADMM

const FloatType = Float64
GUROBI_ENV = nothing                     # problem.jl does not need Gurobi here
include(joinpath(@__DIR__, "..", "..", "problems", "entr_max", "problem.jl"))

const REPO         = abspath(joinpath(@__DIR__, "..", ".."))
const OUT          = joinpath(REPO, "data", "entr_max", "training")
const SIZES        = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
const SHARES       = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [1, 1, 1, 1]
const TAG          = allequal(SHARES) ? "admm20" : "admm20-" * join(SHARES)
const SEED         = 1
const N_TRAIN      = 8000
const N_TEST       = 2000
const PER_INSTANCE = 50
const FIRST_FRAC   = 0.2
const SLME_TOL     = 1e-5
const MAX_ITER     = 1000
const RHO          = 1.0
const S0           = 2000.0

"prox and Moreau envelope of f(x) = x log(x/S0) on x ≥ 1e-9 (the bound of prime_solver_eco_data)."
function prox_env(q::FloatType; ρ = RHO, s = S0)
    y = log(s) + min(ρ*q - 1, 0.0)       # log x; the stationarity condition is convex in y
    for _ in 1:100
        F = y - log(s) + 1 + ρ*(exp(y) - q)
        y -= F / (1 + ρ*exp(y))
        abs(F) < 1e-13 && break
    end
    x = max(exp(y), 1e-9)
    return x, x*log(x/s) + ρ/2*(x - q)^2
end

function instance(rng, n, m)
    A = rand(rng, Uniform(0, 1), m, n)
    b = [sum(A[i, :])/(1.06n) for i in 1:m]      # as data_opt(n, m)
    return data_opt(n, m, A, b, RHO, x -> x*log(x))
end

"sLME-ADMM with the exact prox; returns the prox inputs of every iteration (n × K)."
function admm_inputs(data::data_opt)
    n, m, ρ = data.n, data.m, data.rho
    scale = var_scale(data)
    z = zeros(n + m); w = copy(z); v = copy(z); α = copy(z); β = copy(z)
    M = [sparse(data.A) sparse(I, m, m); sparse(ones(1, n)) spzeros(1, m)]
    proj = AffineProjection(kkt_matrix(M), [scale .* data.b; scale])
    Q = Vector{Vector{FloatType}}()
    for _ in 1:MAX_ITER
        q = v[1:n] + β[1:n]
        push!(Q, q)
        z[1:n] .= first.(prox_env.(q))
        z[n+1:end] .= v[n+1:end] + β[n+1:end]
        v .= proj((z - β + w + α)/2)
        w .= max.(v - α, 1e-9)
        α .+= ρ*(w - v)
        β .+= ρ*(v - z)
        max(maximum(abs.(w - v)), maximum(abs.(v - z))) < SLME_TOL*scale && break
    end
    return reduce(hcat, Q)
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
    X = FloatType[]; E = FloatType[]; G = FloatType[]
    meta = Dict(k => Int[] for k in ("n", "m", "instance", "iteration", "n_iter"))
    for (k, j) in enumerate(size_order(total))
        n, m = SIZES[j]
        Q = admm_inputs(instance(rng, n, m))
        K = size(Q, 2)
        K_keep = ceil(Int, FIRST_FRAC*K)
        pool = unique(i -> Q[i], [CartesianIndex(j, t) for t in 1:K_keep for j in 1:n])
        length(pool) >= PER_INSTANCE || error("instance $k: only $(length(pool)) distinct inputs")
        for idx in pool[randperm(rng, length(pool))[1:PER_INSTANCE]]
            q = Q[idx]
            x, env = prox_env(q)
            push!(X, q); push!(E, env); push!(G, RHO*(q - x))
            for (key, val) in zip(("n", "m", "instance", "iteration", "n_iter"), (n, m, k, idx[2], K))
                push!(meta[key], val)
            end
        end
        @printf("%-5s instance %3d (n=%4d, m=%3d): ADMM %4d it., kept first %3d, %6d distinct q\n",
                split, k, n, m, K, K_keep, length(pool))
    end
    path = joinpath(OUT, "maxEntropy-$(TAG)-rho=$(RHO)-$(split).npz")
    npzwrite(path, merge(Dict("input" => reshape(X, 1, :), "grad" => reshape(G, 1, :), "enve" => E, "rho" => RHO,
                              "seed" => seed, "first_frac" => FIRST_FRAC), meta))
    @printf("wrote %s: %d samples, q in [%.3f, %.3f]\n", path, length(X), minimum(X), maximum(X))
end

mkpath(OUT)
collect_split("train", SEED, N_TRAIN)
collect_split("test",  SEED + 1, N_TEST)
