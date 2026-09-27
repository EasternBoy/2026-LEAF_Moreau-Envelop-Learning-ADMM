# The learned Moreau envelope: an input-convex neural network (ICNN) read from the
# .json written by experiments/*/train.py, and its gradient with preallocated buffers.
# Shared by every problem; the caller defines `const FloatType` before including it.
#
#   rho, mp = load_model("models/<p>/<name>.json")
#   model   = ICNN(mp)
#   mgrad   = gradient_struct(model, nbatch, dim)   # ∇ of the ICNN at a (dim, nbatch) input
#
# The input is laid out as a (dim, nbatch) matrix: entr_max applies a scalar ICNN to a
# batch (dim = 1), mpc and mvee evaluate one vector (nbatch = 1), power_grid a matrix.

@inline function add_bias!(mat, bias)
    cols = size(mat, 2)
    @inbounds @simd for j in 1:cols
        @views mat[:, j] .+= bias
    end
    return mat
end

@inline function activation_sigma!(activ, sigma_buf, preactiv)
    @inbounds @simd for i in eachindex(activ)
        val          = preactiv[i]
        activ[i]     = NNlib.softplus(val)
        sigma_buf[i] = NNlib.σ(val)
    end
    return activ
end

@inline function hadamard!(dest, rhs)
    @inbounds @simd for i in eachindex(dest)
        dest[i] *= rhs[i]
    end
    return dest
end

# dest += m1 * m2, two implementations: BLAS (entr_max, mpc) and an explicit loop
# (power_grid, mvee).  They round differently, so each problem keeps the one it used.
@inline function mul_add!(dest, m1, m2)
    buff = copy(dest)
    mul!(buff, m1, m2)
    dest .+= buff
    return dest
end

@inline function mmul_add_matrix!(dest, src1, src2)
    rows = size(dest, 1)
    cols = size(dest, 2)
    @inbounds for j in 1:cols, i in 1:rows
        @views dest[i, j] += dot(src1[i, :], src2[:, j])
    end
    return dest
end

# ---------------------------------------------------------------------------
function load_model(fname::String)
    data   = JSON3.read(fname)
    vecf64 = (Vector{FloatType} ∘ vec)
    model = (
        U = convert_to_matrix.(data["U"]),
        W = convert_to_matrix.(data["W"]),
        a = vecf64(data["a"]),
        b = vecf64.(data["b"]),
        c = FloatType(data["c"]),
        v = vecf64(data["v"])
    )
    return FloatType(data["rho"]), model
end

function convert_to_matrix(L)
    v = copy(hcat(L...)')
    isempty(v) ? FloatType[] : v
end

struct ICNN_Layer
    U::Matrix{FloatType}
    W::Matrix{FloatType}
    b::Vector{FloatType}
end

struct ICNN
    U0::Matrix{FloatType}
    b0::Vector{FloatType}
    layers::Vector{ICNN_Layer}
    v::Vector{FloatType}
    a::Vector{FloatType}
    c::FloatType
end

"The ICNN of a model read by `load_model`."
ICNN(mp::NamedTuple) = ICNN(mp.U[1], mp.b[1], [ICNN_Layer(mp.U[i], mp.W[i], mp.b[i]) for i in 2:length(mp.U)],
                            mp.v, mp.a, mp.c)

@inbounds function (m::ICNN_Layer)(x::Matrix{FloatType}, z::Matrix{FloatType})
    s = m.W * z + m.U * x .+ m.b
    return map(softplus, s), s  # convex & nondecreasing
end

@inbounds function (m::ICNN)(x::VecOrMat{FloatType})::VecOrMat{FloatType}
    z = softplus.(m.U0 * x .+ m.b0)  # first layer (no state W)
    for layer in m.layers
        z, _ = layer(x, z)
    end
    f = @. m.v'*z + m.a'*x + m.c
    return f
end

# ---------------------------------------------------------------------------
mutable struct gradient_struct{L, K}
    lenlay::Int
    m::ICNN
    s_store::NTuple{L, Matrix{FloatType}}
    σ_store::NTuple{L, Matrix{FloatType}}
    z_store::NTuple{L, Matrix{FloatType}}

    init_grad_x::Matrix{FloatType}
    init_dL_dz::Matrix{FloatType}
    grad_x_buf::Matrix{FloatType}
    dL_curr::Matrix{FloatType}
    dL_next::Matrix{FloatType}
    kernel::K                        # mul_add! or mmul_add_matrix!
end

function gradient_struct(m::ICNN, nbatch::Int, dim::Int; kernel = mul_add!)
    lenlay     = length(m.layers)
    layer_rows = [size(m.U0, 1); [size(layer.W, 1) for layer in m.layers]]
    L          = lenlay + 1

    s_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), L)
    σ_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), L)
    z_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), L)
    init_grad_x = repeat(m.a, 1, nbatch)
    init_dL_dz  = repeat(m.v, 1, nbatch)
    grad_x_buf  = zeros(FloatType, dim, nbatch)
    dL_curr     = zeros(FloatType, size(m.v, 1), nbatch)
    dL_next     = zeros(FloatType, size(m.v, 1), nbatch)

    return gradient_struct(lenlay, m, s_store, σ_store, z_store, init_grad_x, init_dL_dz,
                           grad_x_buf, dL_curr, dL_next, kernel)
end

"∇ICNN at `x`: a (dim, nbatch) matrix, or a vector holding one (entr_max: dim = 1, mpc: nbatch = 1)."
function (obj::gradient_struct)(x::AbstractVecOrMat{FloatType})
    X = x isa AbstractVector && obj.kernel === mul_add! ? reshape(x, size(obj.grad_x_buf, 1), :) : x
    muladd! = obj.kernel

    s_first = obj.s_store[1]
    mul!(s_first, obj.m.U0, X)
    add_bias!(s_first, obj.m.b0)
    activation_sigma!(obj.z_store[1], obj.σ_store[1], s_first)

    for i in 1:obj.lenlay
        layer  = obj.m.layers[i]
        s_next = obj.s_store[i+1]
        mul!(s_next, layer.W, obj.z_store[i])
        muladd!(s_next, layer.U, X)
        add_bias!(s_next, layer.b)
        activation_sigma!(obj.z_store[i+1], obj.σ_store[i+1], s_next)
    end

    copyto!(obj.dL_curr,    obj.init_dL_dz)
    copyto!(obj.grad_x_buf, obj.init_grad_x)

    for i in obj.lenlay:-1:1
        layer = obj.m.layers[i]
        dL_ds = obj.σ_store[i+1]
        hadamard!(dL_ds, obj.dL_curr)
        muladd!(obj.grad_x_buf, layer.U', dL_ds)
        mul!(obj.dL_next, layer.W', dL_ds)
        obj.dL_curr, obj.dL_next = obj.dL_next, obj.dL_curr
    end

    dL_ds_first = obj.σ_store[1]
    hadamard!(dL_ds_first, obj.dL_curr)
    muladd!(obj.grad_x_buf, obj.m.U0', dL_ds_first)

    return x isa AbstractVector ? vec(obj.grad_x_buf) : obj.grad_x_buf
end

# ---------------------------------------------------------------------------
# Chunk size a gradient was built for: its input length for a vector input
# (gradient_struct(model, s_mb, 1) or (model, 1, s_mb)), its number of columns
# for a matrix input (gradient_struct(model, s_mb, dim)).
vector_chunk(g::gradient_struct) = length(g.grad_x_buf)
column_chunk(g::gradient_struct) = size(g.grad_x_buf, 2)

# Evaluate the gradient on a long input in chunks, one chunk per thread.  Each
# chunk has the size the gradients were built for; the last one is aligned to
# the end of the input (it may overlap the previous chunk).

# Chunks of a vector: `local_gradients` built with `gradient_struct(model, s_mb, 1)` or `(model, 1, s_mb)`.
@inbounds function mini_batch(local_gradients::NTuple, batch::Vector{FloatType})
    s_mb      = vector_chunk(local_gradients[1])
    data_size = length(batch)
    n_mb      = div(data_size - 1, s_mb) + 1
    out       = copy(batch)
    @threads for i in 1:n_mb
        r = i == n_mb ? ((data_size - s_mb + 1):data_size) : ((i-1)*s_mb+1:i*s_mb)
        out[r] .= local_gradients[i](batch[r])
    end
    return out
end

# Column chunks of a matrix: `local_gradients` built with `gradient_struct(model, s_mb, dim)`.
@inbounds function mini_batch(local_gradients::NTuple, batch::AbstractMatrix)
    s_mb      = column_chunk(local_gradients[1])
    data_size = size(batch, 2)
    n_mb      = div(data_size - 1, s_mb) + 1
    out       = copy(batch)
    @threads for i in 1:n_mb
        r = i == n_mb ? ((data_size - s_mb + 1):data_size) : ((i-1)*s_mb+1:i*s_mb)
        @views out[:, r] .= local_gradients[i](batch[:, r])
    end
    return out
end
