# Moreau-envelope training data for power_grid, collected in the ADMM loop.
#
#   julia --project=. experiments/power_grid/data_gen_admm.jl          # equal shares of the horizons
#   julia --project=. experiments/power_grid/data_gen_admm.jl 1,3      # shares of N = 96, 192
#
# Writes data/power_grid/training/power_grid_rho=1.0-<tag>-{train,test}.npz, <tag> = admm20 for
# equal shares and admm20-<shares> otherwise: 8000 train and 2000 test samples, split between the
# horizons in proportion to the shares (input (3 × samples), enve, grad (3 × samples), rho, as the
# other data files, plus where each sample came from: N, x0, instance, iteration, n_iter).
#
# * Split like DC3: train and test instances come from two separate RNG streams
#   (Xoshiro(SEED), Xoshiro(SEED + 1)), both different from the table's instances
#   (seed 20262309), so no x0 is shared between train, test and the benchmark.
#   An instance is the table's: the fixed load / PV forecasts of energy_mag(), cut to N, and
#   x0 ~ Uniform(0.25, 0.75) (as generate_table_instances.py); drawn round-robin over the
#   horizons until each has its share.
# * The ADMM is the sMEL-ADMM iteration of problems/power_grid/lme_admm.jl (LME_ADMM_split with
#   dynamics_projection(state_scale = 400), as table.jl) with the exact prox in place of the ICNN,
#   run until its residual is below TOL and the iterate is feasible to FEAS_TOL (as table.jl), or MAX_ITER.
#   If it takes K iterations, only the prox inputs q (columns [m, u, p]) of the first
#   ceil(FIRST_FRAC·K) iterations are kept; duplicates are dropped (the first iterate is q = 0)
#   and PER_INSTANCE samples are drawn from what is left.
# * Labels: the Moreau envelope of the per-step cost eco_mpc(m, u, p) (+ p ≥ 0, as
#   prime_solver_eco_data).  The cost is separable, so the prox is exact: soft thresholds in m
#   and u, and in p the root of ρp³ − ρq p² − r_df·a = 0 on (0, a), found by Newton's method.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Printf, Random, LinearAlgebra, SparseArrays, StaticArrays, NPZ

const FloatType = Float64
GUROBI_ENV = nothing                     # nothing here needs Gurobi
const REPO = abspath(joinpath(@__DIR__, "..", ".."))
cd(REPO)                                 # energy_mag() reads repository-relative CSV paths
include(joinpath(REPO, "problems", "power_grid", "problem.jl"))
include(joinpath(REPO, "problems", "power_grid", "utils.jl"))
include(joinpath(REPO, "problems", "power_grid", "lme_admm.jl"))

const OUT          = joinpath(REPO, "data", "power_grid", "training")
const HORIZONS     = [96, 192]
const SHARES       = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [1, 1]
const TAG          = allequal(SHARES) ? "admm20" : "admm20-" * join(SHARES)
const SEED         = 1
const N_TRAIN      = 8000
const N_TEST       = 2000
const PER_INSTANCE = 50
const FIRST_FRAC   = 0.2
const TOL          = 1e-2                # sMEL-ADMM residual tolerance of table.jl
const FEAS_TOL     = 1e-4                # sMEL-ADMM feasibility tolerance of table.jl (smel_feas_tol)
const MAX_ITER     = 1000
const RHO          = 1.0
const STATE_SCALE  = 400.0               # smel_state_scale of table.jl
const X0_LO, X0_HI = 0.25, 0.75

"MPC data of energy_mag() with horizon N and ρ = RHO (as table.jl)."
function horizon_data(N)
    d = energy_mag()
    c = eco_mpc(d.r_ec, d.r_df, d.r_op, d.η, d.dT, N, d.a)
    return MPCData_eco(d.A, d.B, d.r_ec, d.r_df, d.r_op, d.η, d.BESS, d.dT, d.a, d.x_min, d.x_max, d.x_end_min,
                       d.u_min, d.u_max, d.x0, d.dim, N, d.load_forecast, d.gen_forecast, RHO, c)
end

"prox and Moreau envelope of the per-step cost c(m, u, p) + (p ≥ 0) at q = [m, u, p]."
function prox_env(c::eco_mpc, q::AbstractVector; ρ = RHO)
    c_m = c.r_ec*c.dT                                # slope in m, plus r_op for m > 0
    c_u = c.r_ec*c.dT*(1 - c.η)/(2*sqrt(c.η))        # weight of |u|
    m = q[1] > (c_m + c.r_op)/ρ ? q[1] - (c_m + c.r_op)/ρ : q[1] < c_m/ρ ? q[1] - c_m/ρ : 0.0
    u = sign(q[2])*max(abs(q[2]) - c_u/ρ, 0.0)
    a, r = c.a, c.r_df
    p = if q[3] >= a || ρ*(a - q[3]) <= r/a           # minimiser on p ≥ a, or at the kink p = a
        max(q[3], a)
    else                                              # r(a/p − 1) + ρ/2 (p − q)² on (0, a): convex
        y = a
        for _ in 1:100
            F = ρ*y^3 - ρ*q[3]*y^2 - r*a
            y -= F/(3ρ*y^2 - 2ρ*q[3]*y)
            abs(F) < 1e-12*max(1.0, r*a) && break
        end
        y
    end
    x = [m, u, p]
    return x, c(m, u, p) + ρ/2*sum(abs2, x - q)
end

"sMEL-ADMM with the exact prox; returns the prox inputs of every iteration (3 × N × K)."
function admm_inputs(d::MPCData_eco, proj, x0)
    N = d.N
    load, gen = collect(d.load_forecast[1:N]), collect(d.gen_forecast[1:N])
    z = zeros(FloatType, 4, N); w = copy(z); v = copy(z); α = copy(z); β = copy(z)
    Q = Array{FloatType, 3}(undef, 3, N, 0)
    for _ in 1:MAX_ITER
        b = v .+ β
        Q = cat(Q, b[1:3, :]; dims = 3)
        for k in 1:N
            z[1:3, k] .= first(prox_env(d.cost_func, b[1:3, k]))
        end
        z[4, :] .= b[4, :]                               # no prox for the state
        v .= proj((z .- β .+ w .+ α)./2, x0, load, gen)
        w .= v .- α
        clamp!(@view(w[2, :]), d.u_min, d.u_max)
        clamp!(@view(w[3, :]), 0.0, Inf)
        clamp!(@view(w[4, :]), d.x_min, d.x_max)
        w[4, N] = clamp(w[4, N], max(d.x_min, d.x_end_min), d.x_max)  # terminal bound
        α .+= w .- v
        β .+= v .- z
        residual = max(maximum(abs, w .- v), maximum(abs, v .- z))
        residual < TOL && eco_solution_feasible(d, v, x0, load, gen, FEAS_TOL) && break
    end
    return Q
end

"Instances per horizon for `total` samples, and the round-robin order in which they are drawn."
function horizon_order(total)
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

function collect_split(split, seed, total, cases)
    rng = Xoshiro(seed)
    X = Vector{FloatType}[]; E = FloatType[]; G = Vector{FloatType}[]
    meta = Dict(k => FloatType[] for k in ("N", "x0", "instance", "iteration", "n_iter"))
    for (k, j) in enumerate(horizon_order(total))
        d, proj = cases[j]
        x0 = X0_LO + (X0_HI - X0_LO)*rand(rng)
        Q = admm_inputs(d, proj, x0)
        K = size(Q, 3)
        K_keep = ceil(Int, FIRST_FRAC*K)
        pool = unique(i -> Q[:, i[1], i[2]], [(t, it) for it in 1:K_keep for t in 1:d.N])
        length(pool) >= PER_INSTANCE || error("instance $k: only $(length(pool)) distinct inputs")
        for (t, it) in pool[randperm(rng, length(pool))[1:PER_INSTANCE]]
            q = Q[:, t, it]
            x, env = prox_env(d.cost_func, q)
            push!(X, q); push!(E, env); push!(G, RHO*(q - x))
            for (key, val) in zip(("N", "x0", "instance", "iteration", "n_iter"), (d.N, x0, k, it, K))
                push!(meta[key], val)
            end
        end
        @printf("%-5s instance %3d (N=%3d, x0=%.3f): ADMM %4d it., kept first %3d, %6d distinct q\n",
                split, k, d.N, x0, K, K_keep, length(pool))
    end
    path = joinpath(OUT, "power_grid_rho=$(RHO)-$(TAG)-$(split).npz")
    Xm = reduce(hcat, X)
    npzwrite(path, merge(Dict("input" => Xm, "grad" => reduce(hcat, G), "enve" => E, "rho" => RHO,
                              "seed" => seed, "first_frac" => FIRST_FRAC), meta))
    @printf("wrote %s: %d samples, q in [%s] – [%s]\n", path, length(X),
            join((@sprintf("%.3f", v) for v in minimum(Xm; dims = 2)), ", "),
            join((@sprintf("%.3f", v) for v in maximum(Xm; dims = 2)), ", "))
end

if abspath(PROGRAM_FILE) == @__FILE__
    cases = map(HORIZONS) do N
        d = horizon_data(N)
        (d, dynamics_projection(d; state_scale = STATE_SCALE))
    end
    mkpath(OUT)
    collect_split("train", SEED, N_TRAIN, cases)
    collect_split("test",  SEED + 1, N_TEST, cases)
end
