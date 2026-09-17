using LinearAlgebra
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

include("cost_func.jl")

struct MPCData
    A::VecOrMat{FloatType}
    B::VecOrMat{FloatType}
    nx::Int
    nu::Int
    x_min::FloatType
    x_max::FloatType
    u_min::FloatType
    u_max::FloatType
    x0::Vector{FloatType}
    N::Int
    rho::FloatType
end

# Example from Yalmip
function LTI_2D()
    # --- Model data ---
    A = [2.0 -1.0;
         1.0  0.2]
    B = [1.0; 0.]

    nx = 2           # number of states
    nu = 1           # number of inputs

    # --- MPC data ---
    Q = Matrix{Float64}(I, nx, nx)
    R = 2.0          # scalar
    N = 10            # horizon

    x_max =  5.
    x_min = -5.
    u_max =  1.
    u_min = -1. 

    x0 = [2., 2.]
    rho = 2.

    qp_func = quadratic(Q, R)
    nl_func = logsum(1.0)

    return MPCData(A, B, nx, nu, x_min, x_max, u_min, u_max, x0, N, rho), nl_func
end


struct MPCData_eco
    A::VecOrMat{FloatType}
    B::VecOrMat{FloatType}
    nx::Int
    nu::Int
    x_min::FloatType
    x_max::FloatType
    u_min::FloatType
    u_max::FloatType
    tau::FloatType
    load_track::Vector{FloatType}
    x0::Vector{FloatType}
    N::Int
    rho::FloatType
end

function power_share_star()
    # --- Model data ---
    nx  = 6           # number of power consumers
    nu  = 6
    N   = 24           # horizon

    A = Matrix{FloatType}(I, nx, nx)
    B = Matrix{FloatType}(I, nx, nx)

    x_min = 1e-4
    x_max = Inf
    u_max =  2
    u_min = -2

    a       = [4.; 2.; 3.; 3.; 4.; 2.]   # preferred power consumption level 
    load_tr = [23.; 26.; 25.; 23.; 20.; 16.; 17.; 23.; 23.; 22.; 21.; 20.; 21.; 22.; 21.; 20.; 25.; 30.; 33.; 32.; 28.; 27.; 25.; 21.]    
    x0      = randn(length(a))
    rho     = 1.
    

    τ = maximum(load_tr)
    η₁ = 5
    η₂ = 1

    cost_func = power_share(nx, a, η₁, η₂)

    # available power supply
    # dict = Dict("n"=> n,"N"=> N,"a" => a, "load_tr" => load_tr, "tau" => τ, "x_min" => x_min, "dx_max" => dx_max, "eta1" => η₁, "eta2" => η₂, "x0" => x0)
    return MPCData_eco(A, B, nx, nu, x_min, x_max, u_min, u_max, τ, load_tr, x0, N, rho), cost_func
end


function pick_solver(name, tol = 1e-6)
    #All solvers should share the same tolerance for stopping criteria
    str = lowercase(name)

    if str == "ipopt"      #for LP, QP, NLP
        # println("=============== Selected Ipopt solver =============== ")
        model = Model(Ipopt.Optimizer); set_optimizer_attribute(model, "tol", tol)
    elseif str == "osqp"   #for LP, QP 
        # println("=============== Selected OSQP solver =============== ")
        model = Model(OSQP.Optimizer); set_optimizer_attribute(model, "eps_abs", tol); set_optimizer_attribute(model, "eps_rel", tol)
    elseif str == "gurobi" #for (MI)LP, (MI)QP, or (MI)NLP
        # println("=============== Selected Gurobi solver =============== ")
        if !(@isdefined(GUROBI_ENV))
            GUROBI_ENV = Gurobi.Env()
        end
        model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
        # set_optimizer_attribute(model, "OptimalityTol",  tol)
    elseif str == "clarabel" #for NLP
        # println("=============== Selected clarabel solver =============== ")
        model = Model(Clarabel.Optimizer)
    elseif str == "ecos" #for NLP
        # println("=============== Selected ecos solver =============== ")
        model = Model(ECOS.Optimizer)
    elseif str == "madnlp" #for NLP
        # println("=============== Selected MadNLP solver =============== ")
        model = Model(MadNLP.Optimizer); set_optimizer_attribute(model, "tol", tol)
    else
        error("Specified solver is not supported")
    end
    set_silent(model)
    return model
end