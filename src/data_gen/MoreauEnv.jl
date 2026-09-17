using ParametricOptInterface, MathOptInterface
using JuMP,Ipopt, LinearAlgebra
using NLsolve
using FiniteDiff


const POI = ParametricOptInterface
const MOI = MathOptInterface


include("cost_func.jl")
include("system.jl")


function MoreauEnv(cost_func; rho = 1., )
    nx = dict["nx"]
    nu = dict["nu"]
    nz = nx+nu

    model = pick_solver(name)
    @variable(model, vars[1:nz])
    @variable(model, para[1:nz] in MOI.Parameter.(zeros(nz)))


    J = cost_func(vars[1:nx], vars[nx+1:nz], model) + (rho/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
    @objective(model, Min, J)
    optimize!(model)

    function solver(q::Vector{Float64})
        MOI.set.(model, POI.ParameterValue(), model[:para], q)
        optimize!(model)
        return JuMP.value.(model[:vars]), JuMP.objective_value(model), grad
    end

    return solver
end


function MoreauEnvOnly(q::Vector{Float64}; cost_func = cost_func)
    model = MoreauEnv_solver("Ipopt", cost_func)

    MOI.set.(model, POI.ParameterValue(), model[:para], q) #Without rebuild model
    optimize!(model)

    return JuMP.objective_value(model)
end



function grad_me(q)
    return FiniteDiff.finite_difference_gradient(x-> MoreauEnvOnly(x; cost_func = cost_func), q)
end

function EQ(q,x)
    return q - FiniteDiff.finite_difference_gradient(x -> MoreauEnvOnly(x;cost_func = cost_func), q) - x
end

function f(x;cost_func = cost_func) #Reconstruct f from 
    sol = nlsolve(q-> EQ(q,x), 100*rand(3))
    q = sol.zero
    return MoreauEnvOnly(q;cost_func = cost_func) - 1/(2*ρ)*dot(x-q,x-q)
end


function Eco_MoreauEnv_solver(name::String)
    nu = dict["nu"]

    model = pick_solver(name)
    @variable(model, vars[1:nu])
    @constraint(model, vars .>= 0.0)
    @variable(model, para[1:nu] in MOI.Parameter.(zeros(nu)))


    J = Discomfort(model, vars[1:nu]) + (ρ/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
    @objective(model, Min, J)
    optimize!(model)

    return model
end

function Eco_MoreauEnv(q::Vector{Float64})
    model = Eco_MoreauEnv_solver("Ipopt")

    MOI.set.(model, POI.ParameterValue(), model[:para], q) #Without rebuild model
    optimize!(model)

    return JuMP.value.(model[:vars]), JuMP.objective_value(model)
end

if 1 != 0
    cost_func = L1norm_single
    dict = LTI_2D()
    ρ    = 1.0

    for _ in 1:10
        x = 10rand(2)
        u = 5rand()
        println("Error =",f([x;u];cost_func = cost_func) - cost_func(x, u))
    end
end
