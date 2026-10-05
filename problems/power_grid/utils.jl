using LMEADMM   # src/LMEADMM.jl


function dynamics_projection(mpc_data::MPCData_eco)
    # Euclidean projection onto the equality constraints of the normalized problem,
    # in the variables s = [m̂, û, p̂, x] ∈ R^{4N} (see energy_mag).
    # Rows: N dynamics x_k = A x_{k-1} + B û_k, then N power flow
    # m̂ + (s_u/s_m) û − (s_p/s_m) p̂ = (load − gen)/s_m.  The terminal bound x_N >= x_end_min
    # is an inequality, so it is enforced by the box step of LME_ADMM_split, not here.

    A = mpc_data.A
    B = mpc_data.B
    N = mpc_data.N
    cm, cu, cp = pf_coef(mpc_data)
    s_m = mpc_data.scale[1]
    IN = Matrix{FloatType}(I, N, N)

    Mu = B*IN
    Mx = zeros(N, N)
    Mx[1,1]   = -1.
    for i in 2:N
        Mx[i,i-1] = A
        Mx[i,i]   = -1.
    end

    M = hcat(zeros(N, N), Mu, zeros(N, N), Mx)
    M = vcat(M, hcat(cm*IN, cu*IN, -cp*IN, zeros(N,N)))
    Ms = sparse(M)

    # y = q - M'*(M*M')^(-1)*(M*q-b), equivalent to the full KKT solve.
    F = cholesky(Symmetric(Ms * Ms'))
    query = zeros(FloatType, 4N)
    residual = zeros(FloatType, 2N)
    result = zeros(FloatType, 4, N)

    let F  = F,
        Ms = Ms,
        N  = N,
        A  = A,
        s_m = s_m
        return @inbounds function proj(qm::Matrix{Float64}, init::FloatType, load_fc::Vector{FloatType}, gen_fc::Vector{FloatType})
            for r in 1:4, k in 1:N
                query[(r-1)*N+k] = qm[r,k]
            end
            mul!(residual, Ms, query)
            residual[1] += A * init
            for k in 1:N
                residual[N+k] -= (load_fc[k] - gen_fc[k]) / s_m
            end
            multipliers = F \ residual
            mul!(query, Ms', multipliers, -1.0, 1.0)
            for r in 1:4, k in 1:N
                result[r,k] = query[(r-1)*N+k]
            end
            return result
        end
    end
end



function qp_project_dykstra(q::AbstractVector,
                            A::AbstractMatrix,
                            b::AbstractVector,
                            x_min::AbstractVector,
                            x_max::AbstractVector;
                            max_iter::Int = 5000,
                            tol::Real = 1e-8)

    n = length(q)
    m = size(A, 1)

    # precompute affine projection operator: P_A(x) = x - A' * (AA')⁻¹ (A*x - b)
    AA = A * A'
    F = cholesky(Symmetric(AA))  # assume full row rank
    function proj_affine!(y::AbstractVector, x::AbstractVector)
        tmp = A * x .- b
        tmp = F \ tmp
        y .= x .- A' * tmp
        return y
    end
    
    # box projection
    proj_box!(y, x) = (y .= clamp.(x, x_min, x_max))

    # initialize
    x = copy(q)
    proj_box!(x, x)              # start inside box
    r1 = zeros(eltype(q), n)
    r2 = zeros(eltype(q), n)
    y  = similar(q)
    z  = similar(q)

    for k in 1:max_iter
        x_prev = x

        # Step 1: project onto affine set with correction r1
        z .= x .+ r1
        proj_affine!(y, z)
        r1 .= z .- y

        # Step 2: project onto box with correction r2
        z .= y .+ r2
        proj_box!(x, z)
        r2 .= z .- x

        # convergence
        if norm(x - x_prev) <= tol * max(1.0, norm(x))
            eqres = (m > 0) ? norm(A * x - b) : 0.0
            return (x, k, eqres)
        end
    end

    eqres = (m > 0) ? norm(A * x - b) : 0.0
    return (x, max_iter, eqres)
end



# function gradient_mul_batch(gradient::gradient_struct, data::Matrix{FloatType})
#     rows = size(data, 1)
#     cols = size(data, 2)
#     out  = copy(data)

#     max_col = max(div(100, rows), 1) #100 is recommended by StaticArrays.jl
#     n_mini_batch = div(cols-1, max_col)

#     for i in 1:n_mini_batch
#         @views input = data[:, max_col*(i-1)+1:max_col*i]
#         out[:,max_col*(i-1)+1:max_col*i] .= gradient(SMatrix(input))
#     end
#     @views input = data[:, (max_col*n_mini_batch+1):end]

#     cols > max_col*n_mini_batch+1 ? out[:, (max_col*n_mini_batch):end] .= gradient(SMatrix(input)) :  nothing

#     return out
# end



# function gradient_gen(m::ICNN, nbatch::Int)

#     @inline function add_bias!(mat, bias)
#         rows = size(mat, 1)
#         cols = size(mat, 2)
#         @inbounds for j in 1:cols, i in 1:rows
#             mat[i, j] += bias[i]
#         end
#         return mat
#     end

#     @inline function activation_sigma!(activ, sigma_buf, preactiv)
#         rows = size(preactiv, 1)
#         cols = size(preactiv, 2)
#         @inbounds for j in 1:cols, i in 1:rows
#             val = preactiv[i, j]
#             activ[i, j] = softplus(val)
#             sigma_buf[i, j] = NNlib.σ(val)
#         end
#         return activ
#     end

#     @inline function hadamard!(dest, rhs)
#         rows = size(dest, 1)
#         cols = size(dest, 2)
#         @inbounds for j in 1:cols, i in 1:rows
#             dest[i, j] *= rhs[i, j]
#         end
#         return dest
#     end

#     @inline function mul_add_matrix!(dest, src1, src2)
#         rows = size(dest, 1)
#         cols = size(dest, 2)
#         @inbounds for j in 1:cols, i in 1:rows
#             @views dest[i, j] += dot(src1[i, :], src2[:,j])
#         end
#         return dest
#     end

#     U0 = m.U0
#     b0 = m.b0
#     layers = m.layers
#     lenlay = length(layers)

#     layer_rows = (size(U0, 1), (size(layer.W, 1) for layer in layers)...)
#     store_len = length(layer_rows)

#     s_store = ntuple(i -> MMatrix{layer_rows[i], nbatch, FloatType, layer_rows[i]*nbatch}(undef), store_len)
#     σ_store = ntuple(i -> MMatrix{layer_rows[i], nbatch, FloatType, layer_rows[i]*nbatch}(undef), store_len)
#     z_store = ntuple(i -> MMatrix{layer_rows[i], nbatch, FloatType, layer_rows[i]*nbatch}(undef), store_len)

#     init_grad_x = SMatrix{dim, nbatch, FloatType, dim*nbatch}(repeat(m.a, 1, nbatch))
#     init_dL_dz  = SMatrix{size(m.v, 1), nbatch, FloatType, size(m.v,1)*nbatch}(repeat(m.v, 1, nbatch))

#     grad_x_buf  = MMatrix{dim, nbatch, FloatType, dim*nbatch}(undef)
#     dL_curr     = MMatrix{size(m.v, 1), nbatch, FloatType, size(m.v,1)*nbatch}(undef)
#     dL_next     = MMatrix{size(m.v, 1), nbatch, FloatType, size(m.v,1)*nbatch}(undef)

#     return @inbounds function gradient(x::MMatrix)::MMatrix
#         s_first = s_store[1]
#         mul!(s_first, U0, x)
#         add_bias!(s_first, b0)
#         activation_sigma!(z_store[1], σ_store[1], s_first)

#         for i in 1:lenlay
#             layer  = layers[i]
#             s_next = s_store[i+1]
#             z_prev = z_store[i]

#             mul!(s_next, layer.W, z_prev)
#             mul_add_matrix!(s_next, layer.U, x)
#             add_bias!(s_next, layer.b)
#             activation_sigma!(z_store[i+1], σ_store[i+1], s_next)
#         end

#         copyto!(dL_curr,    init_dL_dz)
#         copyto!(grad_x_buf, init_grad_x)

#         for i in lenlay:-1:1
#             layer = layers[i]
#             dL_ds = σ_store[i+1]
#             hadamard!(dL_ds, dL_curr)
#             mul_add_matrix!(grad_x_buf, layer.U', dL_ds)

#             mul!(dL_next, layer.W', dL_ds)
#             dL_curr, dL_next = dL_next, dL_curr
#         end

#         dL_ds_first = σ_store[1]
#         hadamard!(dL_ds_first, dL_curr)
#         mul_add_matrix!(grad_x_buf, U0', dL_ds_first)

#         return grad_x_buf
#     end
# end
