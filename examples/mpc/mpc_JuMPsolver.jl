function JuMP_solver(name, para_opt::MPCData, tol; verbose::Bool = false)
    model = pick_solver(name, tol)

    nx    = para_opt.nx
    nu    = para_opt.nu
    T     = para_opt.T
    A     = para_opt.A
    B     = para_opt.B
    Q     = para_opt.Q
    Qt    = para_opt.Qt
    R     = para_opt.R    
    x0    = para_opt.x0
    x_min = para_opt.x_min
    x_max = para_opt.x_max
    u_min = para_opt.u_min
    u_max = para_opt.u_max

    @variable(model, x[1:nx, 1:T+1])
    @variable(model, u[1:nu, 1:T])

    # Initial condition
    @constraint(model, x[:, 1] .== x0)

    # Dynamics: x[:,t+1] = A*x[:,t] + B*u[:,t]
    for t in 1:T
        @constraint(model, x[:, t+1] .== A * x[:, t] + B * u[:, t])
    end

    # State bounds 
    for t in 2:T+1
        @constraint(model, x_min .<= x[:, t] .<= x_max)
    end

    # Input bounds
    for t in 1:T
        @constraint(model, u_min .<= u[:, t] .<= u_max)
    end

    # Quadratic cost: sum_t (x'Qx + u'Ru) + x_T'Qt*x_T
    @objective(model, Min,
        sum(x[:, t]' * Q * x[:, t] + u[:, t]' * R * u[:, t] for t in 1:T) +
        x[:, T+1]' * Qt * x[:, T+1]
    )

    optimize!(model)

    if verbose
        @printf("objective value = %5.4f, solve time = %5.3f\n",
                objective_value(model), JuMP.solve_time(model))
    end

    # Return first control action 
    return JuMP.value.(u[:, 1]), JuMP.solve_time(model), objective_value(model)
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
