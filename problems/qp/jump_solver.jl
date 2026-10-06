function JuMP_solver(name, para_opt, tol; verbose::Bool = false)
    scale = var_scale(para_opt)
    model = pick_solver(name, tol)
    n = para_opt.n

    @variable(model, y[1:n])

    @constraint(model, para_opt.A*y .== para_opt.x*scale)
    @constraint(model, para_opt.G*y .<= para_opt.h*scale)

    @objective(model, Min, 0.5 * dot(y, para_opt.Q * y)/scale + dot(para_opt.p, y))

    set_silent(model)
    optimize!(model)
    is_solved_and_feasible(model) || error("$name failed: $(termination_status(model))")

    if verbose == true  @printf("objective value = %5.4f, solve time = %5.3f", objective_value(model), JuMP.solve_time(model)) end

    return JuMP.value.(y)/scale, JuMP.solve_time(model), objective_value(model)/scale
end

function get_benchmark(func_return_time, args::Tuple, n_sample::Int = 100)
    time_arr = zeros(n_sample)
    J = 0.
    for i in 1:n_sample
        _, time_arr[i], J = func_return_time(args...)
        GC.gc()
    end

    return fit(Normal, time_arr[2:end]), J
end
