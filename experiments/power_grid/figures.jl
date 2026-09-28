using Pkg

Pkg.activate(".")
Pkg.instantiate()

using AppleAccelerate
using Printf
using BenchmarkTools
using Profile
using BlockArrays
using SparseArrays
using NNlib
using JSON3
using LinearAlgebra
using StaticArrays
using NPZ
using Plots

const FloatType = Float64
const tol = 1e-4

include("../../problems/power_grid/problem.jl")
include("../../problems/power_grid/jump_solver.jl")
include("../../problems/power_grid/admm.jl")
include("../../problems/power_grid/lme_admm.jl")

rho, mp = load_model("models/power_grid/neco_mpc-rho=1.json")

model = ICNN(
    SMatrix{size(mp.U[1], 1), size(mp.U[1], 2)}(mp.U[1]), 
    SVector{size(mp.b[1], 1)}(mp.b[1]),
    [ICNN_Layer(SMatrix{size(mp.U[i], 1), size(mp.U[i], 2)}(mp.U[i]), SMatrix{size(mp.W[i], 1), size(mp.W[i], 2)}(mp.W[i]), SVector{size(mp.b[i], 1)}(mp.b[i])) for i in 2:length(mp.U)],
    SVector{size(mp.v,1)}(mp.v), 
    SVector{size(mp.a,1)}(mp.a),
    mp.c)

N   = mpc_data.N
dim = mpc_data.dim

mgrad = gradient_struct(model, N, dim; kernel = mmul_add_matrix!)

aux_sol  = dynamics_projection(mpc_data)
admm_sol = LME_ADMM_split(mpc_data, mgrad, aux_sol)

BESSinit = mpc_data.x0
load = SVector{N}(mpc_data.load_forecast[1:N])
gen  = SVector{N}(mpc_data.gen_forecast[1:N])


z = admm_sol(BESSinit, load, gen; verbose = true)
J = get_summary(mpc_data, z)
println("objective_value = $J\n")


plt = plot(layout=(2,2), size=(900,600))
labels = ["(a)", "(b)", "(c)", "(d)"]
units  = ["kW", "kW", "kW", "%"]

for i in 1:4
    # Prepare data and y-limits
    data = (i == 4) ? z[i, :] .* 100 : z[i, :]
    ylims_ = (i == 3) ? (4, 6) : nothing

    plot!(plt[i], t, data,
        xlabel="Time (h)",
        ylabel="",
        xlims=(0,24),
        ylims=ylims_,
        xticks=0:2:24,
        lw=2.,
        legend=false,
        grid=true,
        tickfont=font(12),
        guidefont=font(14))

    # Label positions
    xmin, xmax = extrema(t)
    if ylims_ === nothing
        ymin, ymax = extrema(data)
    else
        ymin, ymax = ylims_
    end
    xpos = xmin + 0.02 * (xmax - xmin)
    ypos = (ymin + ymax) / 2
    annotate!(plt[i], (xpos, ypos, text(labels[i], 14, :bold, :left)))

    # Unit annotation (above left axis)
    ypos_unit = ymax + 0.05 * (ymax - ymin)
    annotate!(plt[i], (xmin + 0.02 * (xmax - xmin), ypos_unit, text(units[i], 12, :left)))
end

# Display and save
display(plt)
savefig(plt, "bess_timeseries.pdf")
println("Figure saved as 'bess_timeseries.pdf'")