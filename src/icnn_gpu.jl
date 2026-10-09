# The learned Moreau envelope on the GPU: the ICNN of src/icnn.jl and its closed-form input
# gradient for a batch of inputs (the columns of a CuMatrix), with the per-layer activations
# (softplus(β t)/β, ELU) and the optional Huber terms.  Not part of the LMEADMM module, so that
# `using LMEADMM` does not load CUDA; a script includes it:
#
#   using CUDA, LMEADMM
#   include("src/icnn_gpu.jl")
#   rho, mp = load_model("models/<p>/<name>.npz")
#   gm  = icnn_gpu(ICNN(mp))                    # weights on the GPU (Float64; T = Float32 also works)
#   g   = gradient_gpu(gm, nbatch)              # buffers for (dim, nbatch) inputs
#   G   = gradient_gpu!(g, X)                   # ∇ICNN at the columns of X::CuMatrix, (dim, nbatch)
#
# Forward pass, layer l = 1..L (z₀ = x):
#   s_l = W_l z_{l-1} + U_l x + b_l   (W_1 = 0),   z_l = σ_l(s_l)
#   f(x) = vᵀ z_L + aᵀ x + c + Σᵢ cᵢ H_δᵢ(Bᵢᵀ x + dᵢ)
# Gradient (backward through the layers, all in matrix products over the batch):
#   e_L = σ_L'(s_L) ⊙ v,   e_l = σ_l'(s_l) ⊙ (W_{l+1}ᵀ e_{l+1}),
#   ∇f(x) = a + Σ_l U_lᵀ e_l + B (c ⊙ clip((Bᵀ x + d) ⊘ δ, −1, 1))

using CUDA, LinearAlgebra

@inline gpu_softplus(t) = max(t, zero(t)) + log1p(exp(-abs(t)))
@inline gpu_sigmoid(t) = one(t) / (one(t) + exp(-t))

"Activation value: kind 0 = softplus(β t)/β, kind 1 = ELU (as LMEADMM.Activation)."
@inline act_value_gpu(kind, β, t) =
    kind == 1 ? (t > zero(t) ? t : expm1(t)) : (β == one(β) ? gpu_softplus(t) : gpu_softplus(β * t) / β)
"Activation derivative."
@inline act_deriv_gpu(kind, β, t) =
    kind == 1 ? (t > zero(t) ? one(t) : exp(t)) : gpu_sigmoid(β * t)

struct ICNNGPU{T}
    U::Vector{CuMatrix{T}}   # U_l, (width_l, dim)
    W::Vector{CuMatrix{T}}   # W_l, (width_l, width_{l-1}); W[1] is unused (0×0)
    b::Vector{CuVector{T}}
    v::CuVector{T}
    a::CuVector{T}
    c::T
    kind::Vector{Int}
    β::Vector{T}
    huber::Union{Nothing, NamedTuple{(:B, :Bt, :d, :c, :δ), Tuple{CuMatrix{T}, CuMatrix{T}, CuVector{T}, CuVector{T}, CuVector{T}}}}
end

"Copies an `ICNN` (src/icnn.jl, from `load_model`) to the GPU in precision T."
function icnn_gpu(m::ICNN; T::Type{<:AbstractFloat} = Float64)
    cu(A) = CuArray{T}(A)
    U = vcat([cu(m.U0)], [cu(l.U) for l in m.layers])
    W = vcat([CuMatrix{T}(undef, 0, 0)], [cu(Matrix(l.W)) for l in m.layers])
    b = vcat([cu(m.b0)], [cu(l.b) for l in m.layers])
    h = m.huber === nothing ? nothing :
        (B = cu(m.huber.B), Bt = cu(Matrix(m.huber.B')), d = cu(m.huber.d), c = cu(m.huber.c), δ = cu(m.huber.δ))
    return ICNNGPU{T}(U, W, b, cu(m.v), cu(m.a), T(m.c), [a.kind for a in m.acts], [T(a.β) for a in m.acts], h)
end

"Preallocated buffers for the gradient at a (dim, nbatch) input."
struct GradientGPU{T}
    m::ICNNGPU{T}
    S::Vector{CuMatrix{T}}   # pre-activations s_l, then σ_l'(s_l)
    Z::Vector{CuMatrix{T}}   # activations z_l
    E::Vector{CuMatrix{T}}   # backward e_l
    H::CuMatrix{T}           # Huber terms (k, nbatch)
    grad::CuMatrix{T}        # (dim, nbatch)
end

function gradient_gpu(m::ICNNGPU{T}, nbatch::Int) where {T}
    widths = [size(U, 1) for U in m.U]
    buf() = [CuMatrix{T}(undef, w, nbatch) for w in widths]
    k = m.huber === nothing ? 0 : length(m.huber.d)
    return GradientGPU{T}(m, buf(), buf(), buf(), CuMatrix{T}(undef, k, nbatch),
                          CuMatrix{T}(undef, size(m.U[1], 2), nbatch))
end

"∇ICNN at the columns of X (dim, nbatch), written into g.grad and returned."
function gradient_gpu!(g::GradientGPU{T}, X::CuMatrix{T}) where {T}
    m = g.m
    L = length(m.U)
    for l in 1:L
        s = g.S[l]
        mul!(s, m.U[l], X)
        l > 1 && mul!(s, m.W[l], g.Z[l-1], one(T), one(T))
        kind, β, b = m.kind[l], m.β[l], m.b[l]
        s .+= b
        g.Z[l] .= act_value_gpu.(kind, β, s)
        s .= act_deriv_gpu.(kind, β, s)          # S[l] now holds σ_l'(s_l)
    end
    g.E[L] .= g.S[L] .* m.v
    g.grad .= m.a
    mul!(g.grad, m.U[L]', g.E[L], one(T), one(T))
    for l in L-1:-1:1
        mul!(g.E[l], m.W[l+1]', g.E[l+1])
        g.E[l] .*= g.S[l]
        mul!(g.grad, m.U[l]', g.E[l], one(T), one(T))
    end
    if m.huber !== nothing
        h = m.huber
        mul!(g.H, h.Bt, X)
        g.H .= h.c .* clamp.((g.H .+ h.d) ./ h.δ, -one(T), one(T))
        mul!(g.grad, h.B, g.H, one(T), one(T))
    end
    return g.grad
end
