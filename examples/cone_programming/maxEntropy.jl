using  AppleAccelerate
using  JuMP
import Clarabel
import LinearAlgebra
import Ipopt
import Gurobi
import MadNLP

using Distributions

import MathOptInterface as MOI

const scale::Int = 2n

if !(@isdefined(GUROBI_ENV))
    GUROBI_ENV = Gurobi.Env() 
end

@kwdef struct data_opt
    n::Int = 1000
    m::Int = 10
    A::Matrix{Float64} = rand(Uniform(0,1), m, n)
    b::Vector{Float64} = [sum(A[i,:])/(1.1*n) for i in 1:m]
    rho::Float64 = 1.
    cost_func::Function = x -> x * log(x)
end

function data_opt(n::Int, m::Int)   
    A   = rand(Uniform(0,1), m, n)
    b   = [sum(A[i,:])/(1.06*n) for i in 1:m]
    rho = 1.
    cost_func = x -> x * log(x)
    return data_opt(n, m, A, b, rho, cost_func)
end

@inbounds function get_objective(data::data_opt, w::Vector{FloatType})
    return sum(data.cost_func.(w))
end
