# Why does sLME-ADMM fail on QP + L1 with a given ICNN?  Compares the model with the exact
# envelope (FISTA prox) on the test split and on the solver's own trajectory.
#   julia --project=. experiments/qp_l1/diagnose.jl models/qp_l1/<model>.npz [instances]
# Reports
#   1. gradient error on the test split, by ADMM iteration (late iterations = near the optimum)
#   2. the largest Hessian eigenvalue of the ICNN: the exact envelope has 0 ≼ ∇²M ≼ ρI, and
#      λ_max > ρ means q - ∇φ(q)/ρ is not the prox of a convex function
#   3. the sLME-ADMM trajectory of the model: gradient error, residual and gap per iteration
#   4. how much systematic gradient error the exact-prox ADMM tolerates
const REPO = abspath(joinpath(@__DIR__, "..", ".."))
using Pkg
Base.active_project() == joinpath(REPO, "Project.toml") || Pkg.activate(REPO; io = devnull)
using Printf, Random, LinearAlgebra, SparseArrays, NPZ, Distributions, Statistics
using LMEADMM

const FloatType = Float64
include(joinpath(REPO, "problems", "qp_l1", "problem.jl"))

const MODEL = length(ARGS) >= 1 ? ARGS[1] : "models/qp_l1/qp_l1-lambda=1-rho=10-128x128-huber100-5000ep.npz"
const N_INST = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
const TOL, FEAS_TOL, GAP, MAX_ITER = 1e-2, 1e-6, 1.0, 1000

rho, mp = load_model(joinpath(REPO, MODEL))
const ρ = rho
const GRAD = gradient_struct(ICNN(mp), 1, 100)
learned(q) = copy(GRAD(q))
exact(q) = ρ * (q - first(prox_env(q; ρ)))
relerr(a, b) = norm(a - b) / max(norm(b), 1e-12)
println("model: $MODEL (ρ = $ρ, widths $(size(mp.U[1], 1)) × $(length(mp.U)))")

# ---- 1. gradient error on the test split, by iteration ----
te = npzread(joinpath(REPO, "data", "qp_l1", "training", "qp_l1-lambda=1.0-rho=10.0-gap=1.0-keep=0.1-test.npz"))
Xq, Gq, it, K = te["input"], te["grad"], te["iteration"], te["n_iter"]
# q = 0 (iteration 1 of every run) has ∇M = 0: left out, as its relative error is undefined
keep = [norm(Gq[:, k]) > 1e-8 for k in axes(Xq, 2)]
err = [relerr(learned(Xq[:, k]), Gq[:, k]) for k in axes(Xq, 2) if keep[k]]
frac = (it ./ K)[keep]
println("\n1. test-split gradient relative error (mean / 90% / max), $(count(.!keep)) samples at q = 0 left out")
for (lo, hi, name) in ((0, 0.1, "first 10% of iterations"), (0.1, 0.5, "10-50%"), (0.5, 0.9, "50-90%"), (0.9, 1.01, "last 10% (near optimum)"))
    e = err[lo .<= frac .< hi]
    isempty(e) || @printf("   %-26s n=%5d  %.3f / %.3f / %.3f\n", name, length(e), mean(e), quantile(e, 0.9), maximum(e))
end

# ---- 2. curvature ----
function hessian_fd(g, q; h = 1e-5)
    H = reduce(hcat, [(g(q + h * e) - g(q - h * e)) / 2h for e in eachcol(Matrix(1.0I, 100, 100))])
    return Symmetric((H + H') / 2)
end
idx = round.(Int, range(1, size(Xq, 2), length = 40))
λl = [extrema(eigvals(hessian_fd(learned, Xq[:, k]))) for k in idx]
λe = [extrema(eigvals(hessian_fd(exact, Xq[:, k]))) for k in idx]
println("\n2. Hessian eigenvalues at 40 test points (exact envelope: within [0, ρ = $ρ])")
@printf("   exact : min %.3f, max %.3f\n", minimum(first, λe), maximum(last, λe))
@printf("   ICNN  : min %.3f, max %.3f, λ_max > ρ at %d / %d points (median λ_max %.3f)\n",
        minimum(first, λl), maximum(last, λl), count(l -> l[2] > ρ, λl), length(λl), median(last.(λl)))

# ---- 3 & 4. sLME-ADMM with a given gradient map ----
inst = npzread(joinpath(REPO, "results", "qp_l1", "table", "instances.npz"))["X"]
Jopt = npzread(joinpath(REPO, "results", "qp_l1", "table", "OSQP.npz"))["J_opt"]

function run_admm(data, Jref, gradmap; trace = false)
    n, m = data.n, data.nineq
    z = zeros(n + m); w = copy(z); v = copy(z); α = copy(z); β = copy(z)
    D = vec(sqrt.(sum(abs2, data.G; dims = 2)))
    M = [sparse(data.A) spzeros(data.neq, m); sparse(data.G ./ D) sparse(I, m, m)]
    proj = AffineProjection(kkt_matrix(M), [data.x; data.h ./ D])
    gap = viol = Inf
    for i in 1:MAX_ITER
        q = v[1:n] + β[1:n]
        g = gradmap(q)
        z[1:n] .= q - g / ρ
        z[n+1:end] .= v[n+1:end] + β[n+1:end]
        v .= proj((z - β + w + α) / 2)
        w[1:n] .= v[1:n] - α[1:n]
        w[n+1:end] .= max.(v[n+1:end] - α[n+1:end], 0.0)
        α .+= w - v; β .+= v - z
        res = max(maximum(abs, w - v), maximum(abs, v - z))
        y = v[1:n]
        gap = 100abs(get_objective(data, y) - Jref) / abs(Jref)
        viol = max(maximum(abs, data.A * y - data.x), maximum(data.G * y - data.h), 0.0)
        trace && i in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) &&
            @printf("   it %4d  |q| %7.2f  grad err %6.3f  residual %.1e  gap %9.3g%%  zeros %3d\n",
                    i, norm(q), relerr(g, exact(q)), res, gap, count(abs.(y) .<= 1e-4))
        res < TOL && viol <= FEAS_TOL && gap <= GAP && return (i, gap, viol)
    end
    return (MAX_ITER, gap, viol)
end

println("\n3. sLME-ADMM with the model, instance 1 (OSQP optimum has $(count(abs.(npzread(joinpath(REPO, "results", "qp_l1", "table", "OSQP.npz"))["W"][1, :]) .<= 1e-4)) zeros)")
run_admm(data_opt(; x = Vector{Float64}(inst[1, :])), Jopt[1], learned; trace = true)

println("\n4. exact gradient + systematic error ε‖∇M(q)‖·d(q), d a fixed smooth unit field; $N_INST instances")
rng = Xoshiro(0); B = randn(rng, 100, 100) / 10
dirfield(q) = (d = tanh.(B * q .+ 0.3); d / norm(d))
for ε in (0.0, 0.01, 0.03, 0.1)
    out = [run_admm(data_opt(; x = Vector{Float64}(inst[k, :])), Jopt[k],
                    q -> (g = exact(q); g + ε * norm(g) * dirfield(q))) for k in 1:N_INST]
    @printf("   ε = %.2f: solved %d / %d, iterations %s, final gap mean %.3g%%\n", ε,
            count(o -> o[1] < MAX_ITER, out), N_INST, string(first.(out)), mean(o -> o[2], out))
end
