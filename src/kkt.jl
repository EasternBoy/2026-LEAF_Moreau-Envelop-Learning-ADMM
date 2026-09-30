# The v-step of sLME-ADMM: the projection onto an affine set {v : M v = b},
#   v = argmin ‖v − q‖²  s.t.  M v = b,
# from the KKT system  [I  Mᵀ; M  −δI] [v; λ] = [q; b],  LDLᵀ-factored once.
# Shared by entr_max, mpc and mvee (power_grid projects in a weighted norm, see its utils.jl).
# Part of the LMEADMM module (src/LMEADMM.jl).
#
#   K = kkt_matrix(M)                 # before the timer, as each problem did
#   P = AffineProjection(K, b)        # factorization: inside or outside the timer, as before
#   v .= P(q)                         # or project!(v, P, q): in place, no allocation

"KKT matrix `[I Mᵀ; M −δI]`; δ > 0 makes it strictly quasi-definite for `ldl`."
function kkt_matrix(M::SparseMatrixCSC; δ::Real = 0.0)
    p, n = size(M)
    dual = δ == 0 ? spzeros(FloatType, p, p) : -δ * sparse(I, p, p)   # no stored zeros when δ = 0
    return [sparse(I, n, n) M'; M dual]
end

struct AffineProjection{T}
    F::T                      # LDLᵀ factorization of the KKT matrix
    RHS::Vector{FloatType}    # [q; b]: b fixed, q written by each call
    n::Int
    sol::Vector{FloatType}    # workspace of project!
end

function AffineProjection(K::SparseMatrixCSC, b::AbstractVector)
    RHS = [zeros(FloatType, size(K, 1) - length(b)); b]
    return AffineProjection(ldl(K), RHS, size(K, 1) - length(b), similar(RHS))
end

function (P::AffineProjection)(q::AbstractVector)
    P.RHS[1:P.n] .= q
    return (P.F \ P.RHS)[1:P.n]
end

"v = P(q) in place: the solve runs in P's workspace."
function project!(v::AbstractVector, P::AffineProjection, q::AbstractVector)
    P.RHS[1:P.n] .= q
    copyto!(P.sol, P.RHS)
    ldiv!(P.F, P.sol)
    @views v .= P.sol[1:P.n]
    return v
end
