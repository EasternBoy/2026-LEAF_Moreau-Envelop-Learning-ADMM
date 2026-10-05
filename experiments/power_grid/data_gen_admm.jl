# Moreau-envelope training data for power_grid, collected in the ADMM loop.
#
#   julia --project=. experiments/power_grid/data_gen_admm.jl              # sMEL-ADMM iterates, equal shares
#   julia --project=. experiments/power_grid/data_gen_admm.jl 1,3          # shares of N = 96, 192
#   julia --project=. experiments/power_grid/data_gen_admm.jl --mode=admm  # regular ADMM iterates (for MEL-ADMM)
#   julia --project=. experiments/power_grid/data_gen_admm.jl --mode=admm --first=40   # keep the first 40% of the iterations
#
# Writes data/power_grid/training/power_grid_rho=1-<tag>-{train,test}.npz, <tag> = sLME_ADMM<first> (sMEL-ADMM)
# or LME_ADMM<first> (--mode=admm), <first> = --first (default 20), for equal shares, with -<shares> appended otherwise:
# 8000 train and 2000 test samples, split between the
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
#   dynamics_projection, as table.jl) with the exact prox in place of the ICNN, on the normalized
#   problem of energy_mag() (m̂ = m/s_m, û = u/s_u, p̂ = p/s_p, cost J/K),
#   run until its residual is below TOL and the iterate is feasible to FEAS_TOL (as table.jl), or MAX_ITER.
#   With --mode=admm the iteration is instead the regular two-block ADMM that MEL-ADMM runs
#   (ADMM_eco_iter / LME_ADMM: prox z of w + α, over-relaxed by GAMMA_ADMM, then the Gurobi QP
#   aux_solver_eco onto all constraints), with the exact prox, run until max|w − z| < TOL and w is
#   feasible to TOL (MEL-ADMM's rule in table.jl), or MAX_ITER; its prox inputs are q = w + α.
#   If it takes K iterations, only the prox inputs q (columns [m, u, p]) of the first
#   ceil(FIRST_FRAC·K) iterations are kept; duplicates are dropped (the first iterate is q = 0)
#   and PER_INSTANCE samples are drawn from what is left.
# * Labels: the Moreau envelope of the normalized per-step cost eco_mpc(m̂, û, p̂) (+ p̂ ≥ 0, as
#   prime_solver_eco_data).  The cost is separable, so the prox is exact: soft thresholds in m
#   and u, and in p the root of ρp³ − ρq p² − r_df·a = 0 on (0, a), found by Newton's method.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)
using Printf, Random, LinearAlgebra, SparseArrays, StaticArrays, NPZ, JuMP, Gurobi

const FloatType = Float64
function gen_arg(name, default)
    i = findfirst(a -> startswith(a, "--$name="), ARGS)
    return i === nothing ? default : String(split(ARGS[i], "=")[2])
end
const MODE = gen_arg("mode", "smel")
const FIRST_PCT = parse(Int, gen_arg("first", "20"))   # percent of the ADMM iterations kept
MODE in ("smel", "admm") || error("--mode must be smel or admm")
GUROBI_ENV = MODE == "admm" ? Gurobi.Env() : nothing   # only the regular ADMM's QP needs Gurobi
const REPO = abspath(joinpath(@__DIR__, "..", ".."))
cd(REPO)                                 # energy_mag() reads repository-relative CSV paths
include(joinpath(REPO, "problems", "power_grid", "problem.jl"))
include(joinpath(REPO, "problems", "power_grid", "utils.jl"))
include(joinpath(REPO, "problems", "power_grid", "lme_admm.jl"))
pick_solver(name, tol = 1e-6, cbs = nothing) = solver_model(name, tol; solvers = @__MODULE__)
include(joinpath(REPO, "problems", "power_grid", "admm.jl"))

const OUT          = joinpath(REPO, "data", "power_grid", "training")
const HORIZONS     = [96, 192]
const POS_ARGS     = filter(a -> !startswith(a, "--"), ARGS)
const SHARES       = length(POS_ARGS) >= 1 ? parse.(Int, split(POS_ARGS[1], ",")) : [1, 1]
const BASE_TAG     = (MODE == "admm" ? "LME_ADMM" : "sLME_ADMM") * string(FIRST_PCT)
const TAG          = allequal(SHARES) ? BASE_TAG : BASE_TAG * "-" * join(SHARES)
const SEED         = 1
const N_TRAIN      = 8000
const N_TEST       = 2000
const PER_INSTANCE = 50
const FIRST_FRAC   = FIRST_PCT / 100
const GAMMA        = 1.6                 # sMEL-ADMM over-relaxation of table.jl (smel_gamma)
const GAMMA_ADMM   = 1.2                 # over-relaxation of ADMM_eco_iter / LME_ADMM (their default γ)
const TOL          = 1e-2                # sMEL-ADMM and MEL-ADMM residual tolerance of table.jl (tol)
const FEAS_TOL     = 1e-4                # sMEL-ADMM feasibility tolerance of table.jl (smel_feas_tol)
const MAX_ITER     = 1000
const RHO          = 1.0
const X0_LO, X0_HI = 0.25, 0.75

"MPC data of energy_mag() with horizon N and ρ = RHO (as table.jl)."
function horizon_data(N)
    d = energy_mag()
    return with_horizon(d, N; rho = RHO)
end

"""
prox and Moreau envelope of the normalized per-step cost ĉ(m̂, û, p̂) = c(s_m m̂, s_u û, s_p p̂)/K
(+ p̂ ≥ 0) at q̂.  ĉ is separable, so each coordinate is the prox of the physical cost term at
s_i q̂_i with penalty ρ K / s_i², divided by s_i: soft thresholds in m and u, and in p the root
of ρ_p p³ − ρ_p q_p p² − r_df·a = 0 on (0, a), found by Newton's method.
"""
function prox_env(c::eco_mpc, q̂::AbstractVector; ρ = RHO)
    s = (c.s_m, c.s_u, c.s_p)
    ρm, ρu, ρp = (ρ*c.K/si^2 for si in s)
    qm, qu, qp = s[1]*q̂[1], s[2]*q̂[2], s[3]*q̂[3]
    c_m = c.r_ec*c.dT                                # slope in m, plus r_op for m > 0
    c_u = c.r_ec*c.dT*(1 - c.η)/(2*sqrt(c.η))        # weight of |u|
    m = qm > (c_m + c.r_op)/ρm ? qm - (c_m + c.r_op)/ρm : qm < c_m/ρm ? qm - c_m/ρm : 0.0
    u = sign(qu)*max(abs(qu) - c_u/ρu, 0.0)
    a, r = c.a, c.r_df
    p = if qp >= a || ρp*(a - qp) <= r/a              # minimiser on p ≥ a, or at the kink p = a
        max(qp, a)
    else                                              # r(a/p − 1) + ρ_p/2 (p − q)² on (0, a): convex
        y = a
        for _ in 1:100
            F = ρp*y^3 - ρp*qp*y^2 - r*a
            y -= F/(3ρp*y^2 - 2ρp*qp*y)
            abs(F) < 1e-12*max(1.0, r*a) && break
        end
        y
    end
    x = [m/s[1], u/s[2], p/s[3]]
    return x, c(x[1], x[2], x[3]) + ρ/2*sum(abs2, x - q̂)
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
        @. z = GAMMA * z + (1 - GAMMA) * v                # over-relaxation, as LME_ADMM_split
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

"Regular two-block ADMM (as MEL-ADMM) with the exact prox; returns the prox inputs of every iteration."
function regular_admm_inputs(d::MPCData_eco, qp, x0)
    N = d.N
    load, gen = collect(d.load_forecast[1:N]), collect(d.gen_forecast[1:N])
    z = zeros(FloatType, 3, N); w = copy(z); α = copy(z)
    Q = Array{FloatType, 3}(undef, 3, N, 0)
    for _ in 1:MAX_ITER
        b = w .+ α
        Q = cat(Q, b; dims = 3)
        for k in 1:N
            z[:, k] .= first(prox_env(d.cost_func, b[:, k]))
        end
        @. z = GAMMA_ADMM * z + (1 - GAMMA_ADMM) * w      # over-relaxation, as LME_ADMM
        w[1, :], w[2, :], w[3, :], _ = qp(z .- α, x0, load, gen)
        α .+= w .- z
        maximum(abs, w .- z) < TOL && eco_solution_feasible(d, w, x0, load, gen, TOL) && break
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
        Q = MODE == "admm" ? regular_admm_inputs(d, proj, x0) : admm_inputs(d, proj, x0)
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
    path = joinpath(OUT, "power_grid_rho=$(isinteger(RHO) ? Int(RHO) : RHO)-$(TAG)-$(split).npz")
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
        (d, MODE == "admm" ? aux_solver_eco("Gurobi", d) : dynamics_projection(d))
    end
    mkpath(OUT)
    collect_split("train", SEED, N_TRAIN, cases)
    collect_split("test",  SEED + 1, N_TEST, cases)
end
