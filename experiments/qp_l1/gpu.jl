# sLME-ADMM on the GPU for QP + L1 (src/icnn_gpu.jl, src/sLME_ADMM_GPU.jl), on the instances and the
# OSQP reference of experiments/qp_l1/table.jl (run that first), with the model of its MODELS.
#   julia --project=. experiments/qp_l1/gpu.jl [--float32] [--gopt=1.0]
# 1. checks the GPU gradient against the CPU gradient (gradient_struct) on the test split;
# 2. solves the instances in batches of 1000, 100 and 1 (the first 100 instances, one at a time)
#    and reports the time per instance: batch wall time / batch size, GPU synchronised.
const REPO = abspath(joinpath(@__DIR__, "..", ".."))
using Pkg
Base.active_project() == joinpath(REPO, "Project.toml") || Pkg.activate(REPO; io = devnull)
using Printf, Random, SparseArrays, LinearAlgebra, NPZ, JuMP, Statistics, Distributions
using CUDA, LMEADMM
cd(REPO)

const FloatType = Float64
include(joinpath(REPO, "problems", "qp_l1", "problem.jl"))
include(joinpath(REPO, "src", "icnn_gpu.jl"))
include(joinpath(REPO, "src", "sLME_ADMM_GPU.jl"))

arg(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=")[2])
const T = "--float32" in ARGS ? Float32 : Float64
const GAP = parse(Float64, arg("gopt", "1.0"))
const MODEL = "models/qp_l1/qp_l1-lambda=1-rho=10-128x128-huber100-5000ep.npz"
const OUT = joinpath(REPO, "results", "qp_l1", "table")
const TOL, FEAS_TOL, MAX_ITER, C_V = 1e-2, 1e-6, 1000, 1e-4

CUDA.functional() || error("CUDA is not functional on this machine")
println("GPU: ", CUDA.name(CUDA.device()), " | precision ", T, " | g_opt ≤ $GAP%")

ρ, mp = load_model(MODEL)
ρ ≈ QP_L1_RHO || error("$MODEL was trained for ρ = $ρ, the problem uses ρ = $QP_L1_RHO")
cpu_model = ICNN(mp)
gm = icnn_gpu(cpu_model; T)

# ---- 1. GPU gradient vs CPU gradient ----
te = npzread(joinpath(REPO, "data", "qp_l1", "training", "qp_l1-lambda=1.0-rho=10.0-gap=1.0-keep=0.1-test.npz"))
Xq = te["input"][:, 1:2000]
gcpu = gradient_struct(cpu_model, 1, 100)
Gc = reduce(hcat, [copy(gcpu(Xq[:, k])) for k in axes(Xq, 2)])
Gg = Array(gradient_gpu!(gradient_gpu(gm, size(Xq, 2)), CuArray{T}(Xq)))
@printf("1. gradient on %d test points: max |GPU - CPU| = %.2e (max |∇| %.2f)\n", size(Xq, 2), maximum(abs, Gg - Gc), maximum(abs, Gc))

# ---- 2. qp_l1 instances ----
inst = npzread(joinpath(OUT, "instances.npz")); Xall = Matrix{Float64}(inst["X"]')   # (neq, N)
Jopt = npzread(joinpath(OUT, "OSQP.npz"))["J_opt"]
N = size(Xall, 2)
prob = qp_gpu_problem(qp_data["A"], qp_data["G"], qp_data["h"]; T)
Qd, pd = CuArray{T}(qp_data["Q"]), CuArray{T}(qp_data["p"])
λ = T(QP_L1_LAMBDA)
objective(Y) = T(0.5) .* sum(Y .* (Qd * Y); dims = 1) .+ (pd' * Y) .+ λ .* sum(abs.(Y); dims = 1)

function solve_batches(cols, bs)
    Y = zeros(length(qp_data["p"]), length(cols)); its = zeros(Int, length(cols)); secs = 0.0
    g = gradient_gpu(gm, bs)
    sLME_ADMM_GPU(prob, g, Xall[:, cols[1:bs]], ρ, objective; J_opt = Jopt[cols[1:bs]], gap = GAP,
                  tol = TOL, feas_tol = FEAS_TOL, max_iter = MAX_ITER)          # warm-up (compile)
    for r in Iterators.partition(eachindex(cols), bs)
        Yb, ib, s = sLME_ADMM_GPU(prob, g, Xall[:, cols[r]], ρ, objective; J_opt = Jopt[cols[r]], gap = GAP,
                                  tol = TOL, feas_tol = FEAS_TOL, max_iter = MAX_ITER)
        Y[:, r] = Yb; its[r] = ib; secs += s
    end
    return Y, its, secs
end

function report(name, cols, Y, its, secs)
    d(k) = data_opt(; x = Xall[:, k])
    J = [get_objective(d(k), Y[:, j]) for (j, k) in enumerate(cols)]
    gap = 100 .* abs.(J .- Jopt[cols]) ./ abs.(Jopt[cols])
    viol = [max(maximum(abs, qp_data["A"] * Y[:, j] - Xall[:, k]), maximum(qp_data["G"] * Y[:, j] - qp_data["h"]), 0.0)
            for (j, k) in enumerate(cols)]
    solved = count((gap .<= GAP) .& (viol .<= C_V) .& (its .< MAX_ITER))
    @printf("| %-34s | %4d/%-4d | %8.3f | %7.1f (%4d) | %.3g (%.3g) | %.1e (%.1e) |\n", name, solved, length(cols),
            1e3secs / length(cols), mean(its), maximum(its), mean(gap), maximum(gap), mean(viol), maximum(viol))
end

println("\n2. sLME-ADMM on the GPU, $N instances (time per instance = batch wall time / batch size)\n")
println("| Run | Solved | Time ms / instance | Iterations mean (max) | Opt. gap % mean (max) | Constr. viol. mean (max) |")
println("|---|---|---|---|---|---|")
for bs in (1000, 100)
    report("batch $bs", collect(1:N), solve_batches(collect(1:N), bs)...)
end
report("batch 1 (first 100 instances)", collect(1:100), solve_batches(collect(1:100), 1)...)
