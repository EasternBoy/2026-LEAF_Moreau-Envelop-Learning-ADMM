# sLME-ADMM on the GPU, batched: one instance per column, all iterates on the device.  The splitting
# of problems/qp/lme_admm.jl for problems of the form
#     min f(y)  s.t.  A y = x,  G y ≤ h          (y ∈ ℝⁿ; A, G, h fixed, x per instance)
# with variables u = [y; s/D] (slack s ≥ 0, Dᵢ = ‖Gᵢ‖), the learned prox z_y = q − ∇ICNN(q)/ρ
# (src/icnn_gpu.jl), and the projection onto {M u = b}, M = [A 0; G/D I], b = [x; h/D], as one
# dense projector applied to the whole batch:  v = P u + Mᵀ(MMᵀ)⁻¹ b,  P = I − Mᵀ(MMᵀ)⁻¹M
# (the same Euclidean projection as the KKT solve of src/kkt.jl).  Not part of the LMEADMM
# module; include it after src/icnn_gpu.jl:
#
#   using CUDA, LMEADMM
#   include("src/icnn_gpu.jl"); include("src/sLME_ADMM_GPU.jl")
#   prob = qp_gpu_problem(A, G, h; T = Float64)
#   out  = sLME_ADMM_GPU(prob, gradient_gpu(icnn_gpu(ICNN(mp)), nbatch), X, ρ, objective; J_opt, gap)
#
# Each instance stops on its own at the first iteration where the sLME_ADMM stopping test holds:
# residual max(‖w − v‖∞, ‖v − z‖∞) < tol, y feasible to feas_tol, and (with J_opt) the gap
# 100|J(y) − J_opt|/|J_opt| ≤ gap.  The batch runs until every instance has stopped or max_iter.

using CUDA, LinearAlgebra

"The fixed data of the problem family on the GPU: the projector and the constraint matrices."
struct QPGPUProblem{T}
    n::Int; m::Int; neq::Int
    P::CuMatrix{T}        # (n+m, n+m)  I − Mᵀ(MMᵀ)⁻¹M
    R::CuMatrix{T}        # (n+m, neq+m)  Mᵀ(MMᵀ)⁻¹: the offset is R b
    A::CuMatrix{T}; G::CuMatrix{T}; h::CuVector{T}; hD::CuVector{T}   # hD = h ./ D
end

function qp_gpu_problem(A::AbstractMatrix, G::AbstractMatrix, h::AbstractVector; T::Type{<:AbstractFloat} = Float64)
    neq, n = size(A); m = size(G, 1)
    D = vec(sqrt.(sum(abs2, G; dims = 2)))
    M = [Matrix{Float64}(A) zeros(neq, m); G ./ D Matrix{Float64}(I, m, m)]
    R = M' / Symmetric(M * M')               # Mᵀ(MMᵀ)⁻¹ in Float64, then converted
    P = I - R * M
    c(X) = CuArray{T}(X)
    return QPGPUProblem{T}(n, m, neq, c(P), c(R), c(A), c(G), c(h), c(h ./ D))
end

"""
    sLME_ADMM_GPU(prob, grad, X, ρ, objective; J_opt = nothing, gap = Inf, tol = 1e-2,
                  feas_tol = 1e-6, max_iter = 1000)

Solves the instances with right-hand sides the columns of X (neq, nbatch) at once.
`grad` is a `gradient_gpu(model, nbatch)`; `objective(Y)` returns the objective of the columns of
Y (n, nbatch) as a (1, nbatch) CuArray.  Returns (Y, iterations, seconds): the solutions
(n, nbatch) on the host, the iteration at which each instance stopped (max_iter if it did not),
and the wall time of the whole batch (setup of the right-hand sides included, GPU synchronised).
"""
function sLME_ADMM_GPU(prob::QPGPUProblem{T}, grad::GradientGPU{T}, X::AbstractMatrix, ρ::Real, objective;
                       J_opt = nothing, gap::Real = Inf, tol::Real = 1e-2, feas_tol::Real = 1e-6,
                       max_iter::Int = 1000) where {T}
    n, m, neq = prob.n, prob.m, prob.neq
    N = size(X, 2)
    @assert size(grad.grad, 2) == N "gradient buffers are for $(size(grad.grad, 2)) columns, X has $N"
    ρT, tolT, feasT, gapT = T(ρ), T(tol), T(feas_tol), T(gap)

    start = time_ns()
    Xd = CuArray{T}(X)
    C = prob.R * vcat(Xd, repeat(prob.hD, 1, N))          # offsets Mᵀ(MMᵀ)⁻¹ b, (n+m, N)
    Jref = J_opt === nothing ? nothing : reshape(CuArray{T}(J_opt), 1, N)

    z = CUDA.zeros(T, n + m, N); w = similar(z) .= 0; v = similar(z) .= 0
    α = similar(z) .= 0; β = similar(z) .= 0; buf = similar(z)
    q = CuMatrix{T}(undef, n, N)
    Ysol = CUDA.zeros(T, n, N)
    done = CUDA.zeros(Bool, 1, N)
    iters = CUDA.fill(Int32(max_iter), 1, N)
    eq = CuMatrix{T}(undef, neq, N); ineq = CuMatrix{T}(undef, m, N)

    for i in 1:max_iter
        # z-update: learned prox on the y block, slacks pass through
        @views q .= v[1:n, :] .+ β[1:n, :]
        g = gradient_gpu!(grad, q)
        @views z[1:n, :] .= q .- g ./ ρT
        @views z[n+1:end, :] .= v[n+1:end, :] .+ β[n+1:end, :]
        # v-update: projection onto {M u = b}
        buf .= (z .- β .+ w .+ α) ./ 2
        mul!(v, prob.P, buf); v .+= C
        # w-update: y free, slacks nonnegative
        w .= v .- α
        @views w[n+1:end, :] .= max.(w[n+1:end, :], zero(T))
        # residual (before the dual updates, as in sLME_ADMM), duals
        res = max.(maximum(abs.(w .- v); dims = 1), maximum(abs.(v .- z); dims = 1))
        α .+= w .- v
        β .+= v .- z
        # stopping test per instance: residual, feasibility of y = v[1:n], gap
        Y = @view v[1:n, :]
        mul!(eq, prob.A, Y); eq .-= Xd
        mul!(ineq, prob.G, Y); ineq .-= prob.h
        viol = max.(maximum(abs.(eq); dims = 1), maximum(ineq; dims = 1), zero(T))
        ok = (res .< tolT) .& (viol .<= feasT)
        if Jref !== nothing
            ok .&= 100 .* abs.(objective(Y) .- Jref) ./ max.(abs.(Jref), eps(T)) .<= gapT
        end
        new = ok .& .!done
        Ysol .= ifelse.(new, Y, Ysol)
        iters .= ifelse.(new, Int32(i), iters)
        done .|= ok
        all(done) && break
    end
    Ysol .= ifelse.(done, Ysol, @view v[1:n, :])          # unfinished instances: the last iterate
    CUDA.synchronize()
    seconds = (time_ns() - start) / 1e9
    return Array{Float64}(Ysol), vec(Array(iters)), seconds
end

# ---------------------------------------------------------------------------
# Hybrid: the ADMM loop on the CPU, only the ICNN gradient on the GPU.  Each iteration copies q
# (n, nbatch) to the device and the gradient back; the projection is the same dense projector,
# applied on the CPU (BLAS).

"The fixed data of `qp_gpu_problem` on the host, for `sLME_ADMM_hybrid`."
struct QPCPUProblem
    n::Int; m::Int; neq::Int
    P::Matrix{Float64}; R::Matrix{Float64}
    A::Matrix{Float64}; G::Matrix{Float64}; h::Vector{Float64}; hD::Vector{Float64}
end

function qp_cpu_problem(A::AbstractMatrix, G::AbstractMatrix, h::AbstractVector)
    neq, n = size(A); m = size(G, 1)
    D = vec(sqrt.(sum(abs2, G; dims = 2)))
    M = [Matrix{Float64}(A) zeros(neq, m); G ./ D Matrix{Float64}(I, m, m)]
    R = M' / Symmetric(M * M')
    return QPCPUProblem(n, m, neq, Matrix(I - R * M), R, Matrix{Float64}(A), Matrix{Float64}(G),
                        Vector{Float64}(h), h ./ D)
end

"""
    sLME_ADMM_hybrid(prob, grad, X, ρ, objective; J_opt, gap, tol, feas_tol, max_iter)

As `sLME_ADMM_GPU`, with every step on the CPU except ∇ICNN (`grad` = `gradient_gpu(model, nbatch)`).
`objective(Y)` takes the host matrix Y (n, nbatch) and returns a vector of length nbatch.
"""
function sLME_ADMM_hybrid(prob::QPCPUProblem, grad::GradientGPU{T}, X::AbstractMatrix, ρ::Real, objective;
                          J_opt = nothing, gap::Real = Inf, tol::Real = 1e-2, feas_tol::Real = 1e-6,
                          max_iter::Int = 1000) where {T}
    n, m, neq = prob.n, prob.m, prob.neq
    N = size(X, 2)
    @assert size(grad.grad, 2) == N
    start = time_ns()
    Xh = Matrix{Float64}(X)
    C = prob.R * vcat(Xh, repeat(prob.hD, 1, N))
    z = zeros(n + m, N); w = zeros(n + m, N); v = zeros(n + m, N); α = zeros(n + m, N); β = zeros(n + m, N)
    buf = similar(z); q = zeros(n, N); g = zeros(n, N)
    qd = CuMatrix{T}(undef, n, N)
    Ysol = zeros(n, N); done = falses(N); iters = fill(max_iter, N)
    eq = zeros(neq, N); ineq = zeros(m, N)
    for i in 1:max_iter
        @views q .= v[1:n, :] .+ β[1:n, :]
        copyto!(qd, T.(q))                                   # host → device
        copyto!(g, Array(gradient_gpu!(grad, qd)))           # gradient, device → host
        @views z[1:n, :] .= q .- g ./ ρ
        @views z[n+1:end, :] .= v[n+1:end, :] .+ β[n+1:end, :]
        buf .= (z .- β .+ w .+ α) ./ 2
        mul!(v, prob.P, buf); v .+= C
        w .= v .- α
        @views w[n+1:end, :] .= max.(w[n+1:end, :], 0.0)
        Y = @view v[1:n, :]
        mul!(eq, prob.A, Y); eq .-= Xh
        mul!(ineq, prob.G, Y); ineq .-= prob.h
        Jy = J_opt === nothing ? nothing : objective(Y)
        for k in 1:N
            done[k] && continue
            res = 0.0
            @inbounds for r in 1:n+m
                res = max(res, abs(w[r, k] - v[r, k]), abs(v[r, k] - z[r, k]))
            end
            viol = max(maximum(abs, view(eq, :, k)), maximum(view(ineq, :, k)), 0.0)
            ok = res < tol && viol <= feas_tol &&
                 (J_opt === nothing || 100abs(Jy[k] - J_opt[k]) / max(abs(J_opt[k]), eps()) <= gap)
            if ok
                done[k] = true; iters[k] = i; Ysol[:, k] .= view(v, 1:n, k)
            end
        end
        α .+= w .- v
        β .+= v .- z
        all(done) && break
    end
    for k in 1:N
        done[k] || (Ysol[:, k] .= view(v, 1:n, k))
    end
    return Ysol, iters, (time_ns() - start) / 1e9
end
