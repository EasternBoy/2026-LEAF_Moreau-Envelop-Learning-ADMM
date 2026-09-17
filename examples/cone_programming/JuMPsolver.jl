function JuMP_solver(name, para_opt, tol, cbs::Union{Nothing, callback_struct} = nothing, init_val::Union{Matrix{FloatType}, Nothing} = nothing; verbose::Bool = false)
    model = pick_solver(name, tol, cbs)
    n = para_opt.n
    m = para_opt.m

    @variable(model,   x[1:n] >= 1e-8)
    @constraint(model, sum(x) == scale)

    @variable(model,   A[1:m, 1:n] in MOI.Parameter.(para_opt.A))
    @variable(model,   b[1:m]      in MOI.Parameter.(para_opt.b))
    @constraint(model, A*x .<= b*scale)

    @objective(model,  Min, sum(x[i] * (log(x[i]) - log(scale)) for i in 1:n))

    set_silent(model)
    optimize!(model)


    if verbose == true  @printf("objective value = %5.4f, solve time = %5.3f", objective_value(model), JuMP.solve_time(model)) end

    return JuMP.value.(x)/scale, JuMP.solve_time(model), objective_value(model)/scale
end

# function Clarabel_solve(data::data_opt, tol::Float64)

#     m, n = data.m, data.n
#     A, b = data.A, data.b

#     model = Model(
#         optimizer_with_attributes(
#             Clarabel.Optimizer,
#             "tol_feas" => tol,
#             "max_iter" => 500
#         )
#     )

#     set_optimizer_attribute(model, "tol_gap_abs", tol)
#     set_optimizer_attribute(model, "tol_gap_rel", tol)


#     set_silent(model)


#     @variable(model, t[1:n])
#     @variable(model, x[1:n])
#     @objective(model, Max, sum(t))
#     @constraint(model, sum(x) == 1)
#     @constraint(model, A * x .<= b)
#     @constraint(model, [i = 1:n], [t[i], x[i], 1] in MOI.ExponentialCone())
#     optimize!(model)
    
    
#     return JuMP.value.(x), solve_time(model), objective_value(model)
# end


function get_benchmark(func_return_time, args::Tuple, n_sample::Int = 100)
    time_arr = zeros(n_sample)
    J = 0.
    for i in 1:n_sample
        _, time_arr[i], J = func_return_time(args...)
        GC.gc()
    end

    return fit(Normal, time_arr[2:end]), J
end