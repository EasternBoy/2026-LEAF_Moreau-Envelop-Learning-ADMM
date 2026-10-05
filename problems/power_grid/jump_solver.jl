function mpc_eco_solver(name, mpc_para, tol,
                        cbs::Union{Nothing, callback_struct} = nothing,
                        init_val::Union{Matrix{FloatType}, Nothing} = nothing;
                        configure = nothing)
    model = pick_solver(name, tol, cbs)

    N    = mpc_para.N
    BESS = mpc_para.BESS
    dT   = mpc_para.dT

    u_max = mpc_para.u_max
    u_min = mpc_para.u_min

    x_max   = mpc_para.x_max
    x_min   = mpc_para.x_min
    x0_init = mpc_para.x0

    load_fc = mpc_para.load_forecast
    gen_fc  = mpc_para.gen_forecast

    cost_func = mpc_para.cost_func

    @variable(model, m[1:N])
    @variable(model, u_min  .<= u[1:N] .<= u_max)
    @variable(model, p[1:N] .>= 0)

    if init_val !== nothing
        JuMP.set_start_value.(m, init_val[1,:])
        JuMP.set_start_value.(u, init_val[2,:])
        JuMP.set_start_value.(p, init_val[3,:])
    end

    @variable(model, x_min  .<= x[0:N] .<= x_max)

    @variable(model, para[1:3, 1:N] in MOI.Parameter.(ones(3, N)))
    @variable(model, x0             in MOI.Parameter.(x0_init))
    @variable(model, load[1:N]      in MOI.Parameter.(load_fc[1:N]))
    @variable(model, generator[1:N] in MOI.Parameter.(gen_fc[1:N]))

    for i in 0:N-1 #Dynamics (normalized û: B = -dT s_u/BESS)
        @constraint(model, x[i+1] == mpc_para.A*x[i] + mpc_para.B*u[i+1])
    end

    @constraint(model, x[N] >= mpc_para.x_end_min) #End constraint
    @constraint(model, x[0] == x0) #Initial state

    cm, cu, cp = pf_coef(mpc_para)
    @constraint(model, cm*m + cu*u - cp*p .== (load - generator)/mpc_para.scale[1]) #Power flow (normalized)

    J = sum(cost_func(m[k], u[k], p[k], model) for k in 1:N)
    @objective(model, Min, J)
    configure !== nothing && configure(model)
    optimize!(model)

    if cbs !== nothing
        cbs.n_iter = Int[]
        cbs.rel_opt_gap = FloatType[]
    end

    function solver(init::FloatType, load::Vector{FloatType}, generator::Vector{FloatType}; verbose = false, return_state = false)
        set_parameter_value.(model[:load], load)
        set_parameter_value.(model[:generator], generator)
        set_parameter_value(model[:x0], init)

        optimize!(model)

        if verbose
            solver_name = lowercase(solution_summary(model).solver)
            J = objective_value(model)
            if solver_name == "ipopt"
                iter_count = barrier_iterations(model)
                println("Objective value = $J after $iter_count iterations")
            elseif solver_name == "madnlp"
                iter_count = barrier_iterations(model)
                println("Objective value = $J after $iter_count iterations")
            end
        end

        vars = vcat(JuMP.value.(model[:m])',  JuMP.value.(model[:u])', JuMP.value.(model[:p])')


        if return_state
            vars = vcat(vars, permutedims([JuMP.value(model[:x][i]) for i in 1:N]))
        end
        return vars, solve_time(model), objective_value(model)
    end

    return solver
end


function get_benchmark(func_return_time, args::Tuple, n_sample::Int = 100)
    time_arr = zeros(n_sample)

    for i in 1:n_sample
        _, time_arr[i] = func_return_time(args...)
        if i % 10 == 0 GC.gc() end
    end

    return fit(Normal, time_arr[2:end])
end
