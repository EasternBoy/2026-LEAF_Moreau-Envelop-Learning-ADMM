# The v-step of sLME-ADMM: the projection onto an affine set {v : M v = b},
#   v = argmin ‖v − q‖²  s.t.  M v = b,
# from the KKT system  [I  Mᵀ; M  −δI] [v; λ] = [q; b],  LDLᵀ-factored once.
# Shared by entr_max, mpc and mvee (power_grid projects in a weighted norm, see its utils.jl).
# The caller loads SparseArrays, LinearAlgebra and LDLFactorizations and defines FloatType.
#
#   K = kkt_matrix(M)                 # before the timer, as each problem did
#   P = AffineProjection(K, b)        # factorization: inside or outside the timer, as before
#   v .= P(q)

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
end

AffineProjection(K::SparseMatrixCSC, b::AbstractVector) =
    AffineProjection(ldl(K), [zeros(FloatType, size(K, 1) - length(b)); b], size(K, 1) - length(b))

function (P::AffineProjection)(q::AbstractVector)
    P.RHS[1:P.n] .= q
    return (P.F \ P.RHS)[1:P.n]
end
