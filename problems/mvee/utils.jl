include("matrix_tools.jl")
const FloatType = Float64
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

@inbounds function (m::ICNN_Layer)(x::Matrix{FloatType}, z::Matrix{FloatType})
    s = m.W * z + m.U * x .+ m.b
    return map(softplus, s), s  # convex & nondecreasing
end


struct ICNN
    U0::Matrix{FloatType}
    b0::Vector{FloatType}
    layers::Vector{ICNN_Layer}
    v::Vector{FloatType}
    a::Vector{FloatType}
    c::FloatType
end

@inbounds function (m::ICNN)(x::Matrix{FloatType})::Matrix{FloatType}
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
    s_store::NTuple
    σ_store::NTuple 
    z_store::NTuple

    init_grad_x::Matrix{FloatType}
    init_dL_dz::Matrix{FloatType}  
    grad_x_buf::Matrix{FloatType} 
    dL_curr::Matrix{FloatType}
    dL_next::Matrix{FloatType}
end

function gradient_struct(m::ICNN, nbatch::Int, dim::Int)

    U0     = m.U0
    layers = m.layers
    lenlay = length(layers)

    layer_rows = hcat(size(U0, 1), [size(layer.W, 1) for layer in layers])
    store_len  = length(layer_rows)

    s_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), store_len)
    σ_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), store_len)
    z_store     = ntuple(i -> zeros(FloatType, layer_rows[i], nbatch), store_len)
    init_grad_x = repeat(m.a, 1, nbatch)
    init_dL_dz  = repeat(m.v, 1, nbatch)

    grad_x_buf  = zeros(FloatType, dim, nbatch)
    dL_curr     = zeros(FloatType, size(m.v, 1), nbatch)
    dL_next     = zeros(FloatType, size(m.v, 1), nbatch)

    return gradient_struct(lenlay, m, s_store, σ_store, z_store, init_grad_x, init_dL_dz, grad_x_buf, dL_curr, dL_next)
end


# @inbounds function mini_batch(local_gradients::NTuple, batch::AbstractMatrix)
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


function (obj::gradient_struct)(x::VecOrMat{FloatType})

    s_first = obj.s_store[1]
    nL = obj.lenlay

    mul!(s_first,      obj.m.U0, x)
    add_bias!(s_first, obj.m.b0)
    activation_sigma!(obj.z_store[1], obj.σ_store[1], s_first)

    for i in 1:nL
        layer  = obj.m.layers[i]
        s_next = obj.s_store[i+1]
        z_prev = obj.z_store[i]

        mul!(s_next, layer.W, z_prev)
        mmul_add_matrix!(s_next, layer.U, x)
        add_bias!(s_next, layer.b)
        activation_sigma!(obj.z_store[i+1], obj.σ_store[i+1], s_next)
    end

    copyto!(obj.dL_curr,    obj.init_dL_dz)
    copyto!(obj.grad_x_buf, obj.init_grad_x)

    for i in nL:-1:1
        layer = obj.m.layers[i]
        dL_ds = obj.σ_store[i+1]
        hadamard!(dL_ds, obj.dL_curr)
        mmul_add_matrix!(obj.grad_x_buf, layer.U', dL_ds)

        mul!(obj.dL_next, layer.W', dL_ds)
        obj.dL_curr, obj.dL_next = obj.dL_next, obj.dL_curr
    end

    dL_ds_first = obj.σ_store[1]
    hadamard!(dL_ds_first, obj.dL_curr)
    mmul_add_matrix!(obj.grad_x_buf, obj.m.U0', dL_ds_first)

    return obj.grad_x_buf
end

# =============== PLOT ELLIPSOID =================

function plot_ellipsoid(sol_admm, sol_mosek, A, obj_GT, obj_admm; x_scale=1.0, y_scale=1.0)

    # ================= Eigen Decomposition ====================
    F = eigen(sol_admm) # Added Symmetric for numerical stability
    λ = F.values
    Q = F.vectors

    Fm = eigen(sol_mosek)
    λm = Fm.values
    Qm = Fm.vectors

    # ================= Semi-axis lengths ======================
    rx, ry, rz = 1 ./ sqrt.(complex.(λ)) 
    rxm, rym, rzm = 1 ./ sqrt.(complex.(λm))
    
    rx, ry, rz = real(rx), real(ry), real(rz)
    rxm, rym, rzm = real(rxm), real(rym), real(rzm)

    # ================= Parameter grid =========================
    n_grid = 30 # Increased slightly for smoother scaled curves
    u = range(0, 2π, length=n_grid)
    v = range(0, π, length=n_grid)

    # ===== ADMM ellipsoid =====
    # 1. Generate unit sphere stretched by semi-axes (Body Frame)
    x0 = [rx*cos(ui)*sin(vj) for ui in u, vj in v]
    y0 = [ry*sin(ui)*sin(vj) for ui in u, vj in v]
    z0 = [rz*cos(vj)         for ui in u, vj in v]

    # 2. Rotate to World Frame AND Apply Scaling
    X = similar(x0); Y = similar(y0); Z = similar(z0)
    for j in axes(x0,2), i in axes(x0,1)
        # Rotate first (Standard Ellipsoid Logic)
        pt = Q * [x0[i,j], y0[i,j], z0[i,j]]
        
        # Scale second (World Frame Scaling)
        X[i,j] = pt[1] * x_scale
        Y[i,j] = pt[2] * y_scale
        Z[i,j] = pt[3]
    end

    # ===== Mosek ellipsoid =====
    xm = [rxm*cos(ui)*sin(vj) for ui in u, vj in v]
    ym = [rym*sin(ui)*sin(vj) for ui in u, vj in v]
    zm = [rzm*cos(vj)         for ui in u, vj in v]

    Xm = similar(xm); Ym = similar(ym); Zm = similar(zm)
    for j in axes(xm,2), i in axes(xm,1)
        pt = Qm * [xm[i,j], ym[i,j], zm[i,j]]
        
        # Scale here as well
        Xm[i,j] = pt[1] * x_scale
        Ym[i,j] = pt[2] * y_scale
        Zm[i,j] = pt[3]
    end

    # ====================== PLOT ==============================
    fig = PyPlot.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    css_GT = "lightskyblue"
    css_LME = "yellow"
    alpha = 0.4
    alpha_LME = 0.4
    
    # Mosek ellipsoid
    ax.plot_surface(Xm, Ym, Zm, color= css_GT, alpha= alpha, linewidth=0, shade=true)

    # ADMM ellipsoid
    ax.plot_surface(X, Y, Z, color= css_LME, alpha= alpha_LME, linewidth=0, shade=true)

    # Scatter samples 
    ax.scatter(A[1,:] .* x_scale, A[2,:] .* y_scale, A[3,:], color="red", s=23, edgecolor="red", label="Samples")

    # ============ GRID STYLING ============
    light_grid_color = (0.9, 0.9, 0.9, 1.0) 
    thin_linewidth   = 0.1 

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"]     = light_grid_color
        axis._axinfo["grid"]["linewidth"] = thin_linewidth
        ax.grid(true) 
    end

    # ============ LEGEND & VIEW ============
    ax.view_init(elev=10, azim=55)
    
    # Axis labels
    ax.set_xlabel("x", fontsize=18,  labelpad=11)
    ax.set_ylabel("y", fontsize=18,  labelpad=11)
    ax.set_zlabel("z", fontsize=18,  labelpad=11)

    # ===================== CUSTOM TICKS (4 per axis) =====================
 # ----------- CLEAN EVENLY-SPACED TICKS -----------
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    # Round limits to 1 decimal place to avoid ugly decimals
    xmin = round(xmin, digits=1)
    xmax = round(xmax, digits=1)
    ymin = round(ymin, digits=1)
    ymax = round(ymax, digits=1)
    zmin = round(zmin, digits=1)
    zmax = round(zmax, digits=1)

    # Now evenly spaced ticks:
    ax.set_xticks(LinRange(xmin, xmax, 3))
    ax.set_yticks(LinRange(ymin, ymax, 3))
    ax.set_zticks(LinRange(zmin, zmax, 3))

    # 1-decimal formatting
    fmt = PyPlot.matplotlib.ticker.FormatStrFormatter("%.1f")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.zaxis.set_major_formatter(fmt)


    # Tick sizing
    ax.tick_params(axis="both", which="major", labelsize=16, pad = 6)

    # Legend Construction
    mpatches = PyPlot.matplotlib.patches
    mlines   = PyPlot.matplotlib.lines

    label_GT  = @sprintf("\$J^*\$ = %.2f", obj_GT)
    label_LME = @sprintf("\$J^*\$ = %.2f", obj_admm)

    proxy_GT = mpatches.Rectangle((0,0), 1, 1,
                                fc=css_GT, alpha=alpha,
                                edgecolor="black", linewidth=0.5,
                                label=label_GT)

    proxy_LME = mpatches.Rectangle((0,0), 1, 1,
                                fc=css_LME, alpha=alpha_LME,
                                edgecolor="black", linewidth=0.5,
                                label=label_LME)

    ax.legend(handles=[proxy_GT, proxy_LME], loc="upper left", fontsize=18, frameon=true, framealpha=1.0)
    
    try
        ax.set_box_aspect((1,1,1))
    catch
    end

    PyPlot.show()
    fig.savefig("ellipsoid_scaled.png", dpi=1000, bbox_inches="tight", pad_inches=0)
    
    return fig
end