function logdet_solver(name::String, data)
    model = pick_solver(name) 
    set_silent(model)  

    m   = data.m
    n   = data.n

    # --- Decision variable: X ∈ Sⁿ, PSD ---
    @variable(model, X[1:n, 1:n], Symmetric)
    
    # --- Enforce X ≽ eps * I (diagonal ≥ eps) ---
    for i in 1:n
        @constraint(model, X[i, i] >= 1e-9)
    end

    @constraint(model, c[i=1:m], dot(data.A[:,i], X, data.A[:,i]) .<= 1.)

    @variable(model, t)

    # --- FIX: Pass 1.0 directly into the cone instead of a variable 'u' ---
    @constraint(
        model,
        [t; 1.; triangle_vec(X)] in MOI.LogDetConeTriangle(n)
    )

    @objective(model, Max, t)

    optimize!(model)

    return @inbounds function solver(data::data_opt)

        # --- Constraints: a_i' * X * a_i ≤ 1 for i = 1..m ---
        delete(model, c)
        unregister(model, :c)

        @constraint(model, c[i=1:m], dot(data.A[:,i], model[:X], data.A[:,i]) .<= 1.)

        optimize!(model)

        return value.(model[:X]), solution_summary(model).solve_time, data.cost_func(value.(model[:X]))
    end
    return solver

end