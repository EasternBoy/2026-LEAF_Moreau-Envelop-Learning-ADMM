const FloatType = Float64
using LMEADMM   # src/LMEADMM.jl


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