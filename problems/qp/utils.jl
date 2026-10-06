using LMEADMM   # src/LMEADMM.jl

# ICNN_MODEL, when defined before this file is included, picks another model
rho, mp = load_model(@isdefined(ICNN_MODEL) ? ICNN_MODEL : "models/qp/qp-julia-moreau-rho=1.npz")
# Keep equal hidden widths for gradient_struct's reusable backward buffers.
function prune_qp_model(mp)
    length(mp.U) == 2 || return mp
    size(mp.U[1], 1) == size(mp.U[2], 1) || return mp
    keep2 = findall(!iszero, mp.v)
    keep1 = findall(vec(any(!iszero, mp.W[2][keep2, :]; dims = 1)))
    width = max(length(keep1), length(keep2), 1)
    append!(keep1, setdiff(axes(mp.U[1], 1), keep1)[1:width-length(keep1)])
    append!(keep2, setdiff(axes(mp.U[2], 1), keep2)[1:width-length(keep2)])
    return merge(mp, (U = [mp.U[1][keep1, :], mp.U[2][keep2, :]],
                      W = [mp.W[1][keep1, :], mp.W[2][keep2, keep1]],
                      b = [mp.b[1][keep1], mp.b[2][keep2]], v = mp.v[keep2]))
end

model = ICNN(prune_qp_model(mp))

@assert rho ≈ qp_data["rho"][]
