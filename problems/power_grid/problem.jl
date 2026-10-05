using Ipopt  
using OSQP   # Operator-splitting QP solver (good for large problems)
using HiGHS
using SCS
using Gurobi
using MadNLP
using ParameterJuMP
using JuMP
using Clarabel
using ECOS
using CSV, DataFrames


#========== quadratic  ========#
# Per-step cost of the *normalized* problem: the variables are m̂ = m/s_m, û = u/s_u,
# p̂ = p/s_p (physical m, u, p in kW) and the cost is divided by K, so that the
# variables are O(1) and ADMM can use ρ = 1.  c(m̂, û, p̂) = c_phys(s_m m̂, s_u û, s_p p̂)/K.
struct eco_mpc
    r_ec::FloatType
    r_df::FloatType
    r_op::FloatType
    η::FloatType
    dT::FloatType
    N::Int
    a::FloatType
    s_m::FloatType
    s_u::FloatType
    s_p::FloatType
    K::FloatType
end


function discomfort(p, a)
    return a/p - 1
end


function (obj::eco_mpc)(m, u, p, model = nothing)   # normalized m̂, û, p̂
    r_ec  = obj.r_ec
    r_df  = obj.r_df
    r_op  = obj.r_op
    η     = obj.η
    dT    = obj.dT
    a = obj.a
    s_m, s_u, s_p, K = obj.s_m, obj.s_u, obj.s_p, obj.K

    if model !== nothing   # epigraph variables in normalized units
        su = @variable(model)
        sm = @variable(model)
        sd = @variable(model)

        @constraint(model,  u <= su)
        @constraint(model, -u <= su)

        @constraint(model,  sm >= 0)
        @constraint(model,  sm >= m)

        @constraint(model,  sd >= 0)
        @constraint(model,  sd >= discomfort(s_p*p, a))

        return (r_ec*dT*(s_m*m + (1-η)/(2*sqrt(η))*s_u*su) + r_op*s_m*sm + r_df*sd)/K
    else #just for computation
        return (r_ec*dT*(s_m*m + (1-η)/(2*sqrt(η))*s_u*abs(u)) + r_op*s_m*max(m, 0) +
                r_df*max(discomfort(s_p*p, a), 0))/K
    end
end


const POWER_GRID_K = 2.0e4   # cost scale K of the normalized problem (Ĵ = J/K), tuned on held-out x0

#========= Parameters  ========#
struct MPCData_eco
    A::FloatType
    B::FloatType
    r_ec::FloatType
    r_df::FloatType
    r_op::FloatType
    η::FloatType
    BESS::FloatType
    dT::FloatType
    a::FloatType
    x_min::FloatType
    x_max::FloatType
    x_end_min::FloatType   # terminal constraint x_N >= x_end_min
    u_min::FloatType       # normalized (u_min/s_u)
    u_max::FloatType       # normalized (u_max/s_u)
    x0::FloatType
    dim::Int
    N::Int
    load_forecast::SVector
    gen_forecast ::SVector
    rho::FloatType
    cost_func::eco_mpc
    scale::NTuple{3, FloatType}   # (s_m, s_u, s_p): m = s_m m̂, u = s_u û, p = s_p p̂
    K::FloatType                  # cost scale: J = K Ĵ
end

# Power flow u + m + gen − load − p = 0 in normalized variables, divided by s_m:
#     m̂ + (s_u/s_m) û − (s_p/s_m) p̂ = (load − gen)/s_m
pf_coef(d) = (1.0, d.scale[2]/d.scale[1], d.scale[3]/d.scale[1])
pf_rhs(d, load, gen) = (load .- gen) ./ d.scale[1]
"Physical [m, u, p(, x)] (kW, kW, kW, SOC) of a normalized solution."
to_physical(d, v) = vcat(d.scale[1] .* v[1:1, :], d.scale[2] .* v[2:2, :], d.scale[3] .* v[3:3, :], v[4:end, :])

function energy_mag(; K = POWER_GRID_K)
    # --- Model data ---
    dT    = 0.25
    A     = 1.
    BESS  = 500
    r_ec  = 0.1
    r_df  = 10.
    r_op  = 19.19
    eta   = 0.8
    a     = 50.
    x_min = 0.2
    x_max = 0.8
    x_end_min = 0.5
    u_max = 700
    u_min = -700
    x0    = 0.5
    dim   = 3
    N     = 96
    rho   = 1 

    path_power_gen_data  = "data/power_grid/micro_grid/PV_48h_15-min_150kW_San_Diego.csv"
    path_power_load_data = "data/power_grid/micro_grid/load_15min_max100kW_SanDiego_Building.csv"

    gf = CSV.read(path_power_gen_data, DataFrame)[:,2]
    gen_forecast = SVector{size(gf,1)}(gf)
    lf = CSV.read(path_power_load_data, DataFrame)[:,2]
    load_forecast = SVector{size(lf,1)}(lf) 

    # --- Normalization: power variables by their maximum, cost by K ---
    s_u = FloatType(u_max)                    # |u| ≤ u_max
    s_p = FloatType(maximum(load_forecast))   # delivered power ~ load
    s_m = s_p + s_u                           # grid import ≤ load + battery
    B   = - dT/BESS * s_u                     # x_k = A x_{k-1} + B û_k

    cost_func = eco_mpc(r_ec, r_df, r_op, eta, dT, N, a, s_m, s_u, s_p, K)

    # available power supply
    return MPCData_eco(A, B, r_ec, r_df, r_op, eta, BESS, dT, a, x_min, x_max, x_end_min, u_min/s_u, u_max/s_u,
                       x0, dim, N, load_forecast, gen_forecast, rho, cost_func, (s_m, s_u, s_p), FloatType(K))
end

"Data of `d` with horizon N and ADMM penalty rho (the cost and normalization are kept)."
function with_horizon(d::MPCData_eco, N::Int; rho = d.rho)
    c = d.cost_func
    c = eco_mpc(c.r_ec, c.r_df, c.r_op, c.η, c.dT, N, c.a, c.s_m, c.s_u, c.s_p, c.K)
    return MPCData_eco(d.A, d.B, d.r_ec, d.r_df, d.r_op, d.η, d.BESS, d.dT, d.a, d.x_min, d.x_max, d.x_end_min,
                       d.u_min, d.u_max, d.x0, d.dim, N, d.load_forecast, d.gen_forecast, FloatType(rho), c, d.scale, d.K)
end
