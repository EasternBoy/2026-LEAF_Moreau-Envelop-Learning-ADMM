using JuMP
using LinearAlgebra
using Ipopt
using OSQP
using Gurobi
using Clarabel
using Random
using Distributions
using ControlSystems

const FloatType = Float64

# ── Cost function ──

struct CostMPC
    Q::Matrix{FloatType}
    Qt::Matrix{FloatType}
    R::Matrix{FloatType}
end

function (obj::CostMPC)(x::AbstractVector, u::AbstractVector)
    obj_value = 0.
    for i in eachindex(u)
        obj_value += dot(x[i], obj.Q, x[i]) + dot(u[i], obj.R, u[i])
    end
    obj_value += dot(x[end], obj.Qt, x[end])
    return obj_value
end

# ── Problem data ──

struct MPCData
    A::Matrix{FloatType}
    B::Matrix{FloatType}
    Q::Matrix{FloatType}
    Qt::Matrix{FloatType}
    R::Matrix{FloatType}
    rho::FloatType
    x_min::Vector{FloatType}
    x_max::Vector{FloatType}
    u_min::Vector{FloatType}
    u_max::Vector{FloatType}
    nx::Int
    nu::Int
    T::Int                      # horizon
    x0::Vector{FloatType}       # initial state
    cost_func::CostMPC
end

function MPCData()
    seed = 2026
    rng  = Random.MersenneTwister(seed)

    nx         = 20
    nu         = nx ÷ 2
    T          = 10
    rho        = 1.0

    delta = 0.01 * randn(rng, nx, nx)
    Ix = Matrix{FloatType}(I, nx, nx)
    Iu = Matrix{FloatType}(I, nu, nu)
    A = Ix + delta

    B = randn(rng, nx, nu)

    q = zeros(nx)
    num_nonzero = round(Int, 0.7*nx)
    idx = randperm(rng, nx)[1:num_nonzero]
    q[idx] .= rand(rng, num_nonzero) .* 10
    Q = diagm(q)
    R = 0.1 * Iu

    # Qt = 0.01*dare(A, B, Q, R)
    Qt = Q

    x_max = rand(rng, nx) .+ 1
    x_min = -x_max
    u_max = rand(rng, nu) .* 0.1
    u_min = -u_max

    x0 = 0.5*x_min .+ rand(rng, nx) .* (0.5*x_max .- 0.5*x_min)
    cost_func = CostMPC(Q, Qt, R)
    return MPCData(A, B, Q, Qt, R, rho, x_min, x_max, u_min, u_max, nx, nu, T, x0, cost_func)
end


function new_instance(para_opt::MPCData)
    x0 = 0.5*para_opt.x_min .+ rand(length(para_opt.x_min)) .* (0.5*para_opt.x_max .- 0.5*para_opt.x_min)
    return MPCData(para_opt.A, para_opt.B, para_opt.Q, para_opt.Qt, para_opt.R,
                   para_opt.rho, para_opt.x_min, para_opt.x_max,
                   para_opt.u_min, para_opt.u_max,
                   para_opt.nx, para_opt.nu, para_opt.T, x0, para_opt.cost_func)
end


function pick_solver(name, tol=1e-8)
    str = lowercase(name)

    if str == "ipopt"
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "tol", tol)

    elseif str == "osqp"
        model = Model(OSQP.Optimizer)
        set_optimizer_attribute(model, "eps_abs", tol)
        set_optimizer_attribute(model, "eps_rel", 0)
        set_optimizer_attribute(model, "scaling", 0)

    elseif str == "gurobi"
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "Threads",        Threads.nthreads())
        set_optimizer_attribute(model, "OptimalityTol",  tol)
        set_optimizer_attribute(model, "FeasibilityTol", tol)
        set_optimizer_attribute(model, "BarConvTol",     tol)
        set_optimizer_attribute(model, "Method",         2)
        set_optimizer_attribute(model, "ScaleFlag",      0)

    elseif str == "clarabel"
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "tol_gap_abs", tol)
        set_optimizer_attribute(model, "tol_gap_rel", 0)

    else
        error("Unsupported solver: $name")
    end

    set_silent(model)
    return model
end
