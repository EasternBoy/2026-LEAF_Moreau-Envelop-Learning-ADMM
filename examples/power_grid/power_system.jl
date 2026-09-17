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
struct eco_mpc
    r_ec::FloatType
    r_df::FloatType
    r_op::FloatType
    η::FloatType
    dT::FloatType
    N::Int
    a::FloatType
end


function discomfort(p, a)
    return a/p - 1
end


function (obj::eco_mpc)(m, u, p, model = nothing)
    r_ec  = obj.r_ec
    r_df  = obj.r_df
    r_op  = obj.r_op
    η     = obj.η
    dT    = obj.dT
    a = obj.a

    if model !== nothing
        su = @variable(model)
        sm = @variable(model)
        sd = @variable(model)

        @constraint(model,  u <= su)
        @constraint(model, -u <= su)

        @constraint(model,  sm >= 0)
        @constraint(model,  sm >= m)

        @constraint(model,  sd >= 0)
        @constraint(model,  sd >= discomfort(p, a))

        return r_ec*dT*(m + (1-η)/(2*sqrt(η))*su) + r_op*sm + r_df*sd
    else #just for computation
        return r_ec*dT*(m + (1-η)/(2*sqrt(η))*abs(u)) + r_op*max(m, 0) + r_df*max(discomfort(p, a), 0)
    end
end


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
    u_min::FloatType
    u_max::FloatType
    x0::FloatType
    dim::Int
    N::Int
    load_forecast::SVector
    gen_forecast ::SVector
    rho::FloatType
    cost_func::eco_mpc
end

function energy_mag()
    # --- Model data ---
    dT    = 0.25
    A     = 1.
    BESS  = 500
    B     = - dT/BESS
    r_ec  = 0.1
    r_df  = 10.
    r_op  = 19.19
    eta   = 0.8
    a     = 50.
    x_min = 0.2
    x_max = 0.8
    u_max = 700
    u_min = -700
    x0    = 0.5
    dim   = 3
    N     = 96
    rho   = 1 

    path_power_gen_data  = "data/micro_grid/PV_48h_15-min_150kW_San_Diego.csv"
    path_power_load_data = "data/micro_grid/load_15min_max100kW_SanDiego_Building.csv"

    gf = CSV.read(path_power_gen_data, DataFrame)[:,2]
    gen_forecast = SVector{size(gf,1)}(gf)
    lf = CSV.read(path_power_load_data, DataFrame)[:,2]
    load_forecast = SVector{size(lf,1)}(lf) 

    cost_func = eco_mpc(r_ec, r_df, r_op, eta, dT, N, a)

    # available power supply
    return MPCData_eco(A, B, r_ec, r_df, r_op, eta, BESS, dT, a, x_min, x_max, u_min, u_max, x0, dim, N, load_forecast, gen_forecast, rho, cost_func)
end