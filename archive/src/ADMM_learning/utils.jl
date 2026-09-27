include("matrix_tools.jl")

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


rho, mp = load_model("examples/entr_max/mEntropy-rho=1.json")

model = ICNN(
    mp.U[1], 
    mp.b[1],
    [ICNN_Layer(mp.U[i], mp.W[i], mp.b[i]) for i in 2:length(mp.U)],
    mp.v, 
    mp.a,
    mp.c)

const lenlay::Int = length(model.layers)



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

mutable struct gradient_struct
    lenlay::Int
    m::ICNN
    s_store::Tuple{Vararg{Matrix{FloatType}, lenlay+1}}
    σ_store::Tuple{Vararg{Matrix{FloatType}, lenlay+1}}
    z_store::Tuple{Vararg{Matrix{FloatType}, lenlay+1}}

    init_grad_x::Matrix{FloatType}
    init_dL_dz::Matrix{FloatType}  
    grad_x_buf::Matrix{FloatType} 
    dL_curr::Matrix{FloatType}
    dL_next::Matrix{FloatType}
end

function gradient_struct(m::ICNN, nbatch::Int, dim::Int)

    U0     = m.U0
    layers = m.layers

    layer_rows = hcat(size(U0, 1), [size(layer.W, 1) for layer in layers])

    s_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), lenlay+1)
    σ_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), lenlay+1)
    z_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), lenlay+1)
    init_grad_x = repeat(m.a, 1, nbatch)
    init_dL_dz  = repeat(m.v, 1, nbatch)

    grad_x_buf  = zeros(FloatType, dim, nbatch)

    dL_curr     = zeros(FloatType, size(m.v, 1), nbatch)
    dL_next     = zeros(FloatType, size(m.v, 1), nbatch)

    return gradient_struct(lenlay, m, s_store, σ_store, z_store, init_grad_x, init_dL_dz, grad_x_buf, dL_curr, dL_next)
end


# @inbounds function mini_batch(local_gradients::NTuple, batch::Matrix{FloatType})
#     data_size = size(batch, 2)
#     n_mb      = div(data_size -1, s_mb) + 1
#     out       = copy(batch)

#     @threads for i in 1:n_mb
#         if i == n_mb
#             @views out[:, (data_size - s_mb + 1):data_size] .= local_gradients[i](batch[:, (data_size - s_mb + 1):data_size])
#         else
#             @views out[:,(i-1)*s_mb+1:i*s_mb] .= local_gradients[i](batch[:,(i-1)*s_mb+1:i*s_mb])
#         end
#     end
#     return out
# end


function (obj::gradient_struct)(x::Vector{FloatType})

    s_first = obj.s_store[1]
    nL = obj.lenlay

    mul!(s_first, obj.m.U0, x')
    add_bias!(s_first, obj.m.b0)
    activation_sigma!(obj.z_store[1], obj.σ_store[1], s_first)

    for i in 1:nL
        layer  = obj.m.layers[i]
        s_next = obj.s_store[i+1]
        z_prev = obj.z_store[i]

        mul!(s_next, layer.W, z_prev)
        mul_add!(s_next,  layer.U, x')
        add_bias!(s_next, layer.b)
        activation_sigma!(obj.z_store[i+1], obj.σ_store[i+1], s_next)
    end

    copyto!(obj.dL_curr,    obj.init_dL_dz)
    copyto!(obj.grad_x_buf, obj.init_grad_x)

    for i in nL:-1:1
        layer = obj.m.layers[i]
        dL_ds = obj.σ_store[i+1]
        hadamard!(dL_ds, obj.dL_curr)
        mul_add!(obj.grad_x_buf, layer.U', dL_ds)

        mul!(obj.dL_next, layer.W', dL_ds)
        obj.dL_curr, obj.dL_next = obj.dL_next, obj.dL_curr
    end

    dL_ds_first = obj.σ_store[1]
    hadamard!(dL_ds_first, obj.dL_curr)
    mul_add!(obj.grad_x_buf, obj.m.U0', dL_ds_first)

    return vec(obj.grad_x_buf)
end

@inline function LU_decomp(x)
    return lu(x)
end


@inbounds function mini_batch(local_gradients::NTuple, batch::Vector{FloatType})
    data_size = length(batch)
    n_mb      = div(data_size - 1, s_mb) + 1
    out       = copy(batch)

    @threads for i in 1:n_mb
        if i == n_mb
            out[(data_size - s_mb + 1):data_size] .= local_gradients[i](batch[(data_size - s_mb + 1):data_size])
        else
            out[(i-1)*s_mb+1:i*s_mb] .= local_gradients[i](batch[(i-1)*s_mb+1:i*s_mb])
        end
    end
    return out
end