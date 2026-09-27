using JuMP
import Clarabel
import LinearAlgebra
import MathOptInterface as MOI

const FloatType = Float64

include("../../problems/entr_max/problem.jl")
data = data_opt(n)


function Clarabel_solve(data::data_opt)

    m, n = data.m, data.n
    A, b = data.A, data.b

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, t[1:n])
    @variable(model, x[1:n])
    @objective(model, Max, sum(t))
    @constraint(model, sum(x) == scale)
    @constraint(model, A * x .<= b*scale)
    @constraint(model, [i = 1:n], [t[i], x[i], 1] in MOI.ExponentialCone())
    optimize!(model)
    
    return JuMP.value.(x)/scale, solve_time(model), objective_value(model)/scale
end