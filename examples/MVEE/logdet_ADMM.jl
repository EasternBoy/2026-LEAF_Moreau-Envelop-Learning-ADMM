using Convex
using MathOptInterface
using ParametricOptInterface

const MOI = MathOptInterface
const POI = ParametricOptInterface

function ADMM_logdet_iter(mpc_para::data_opt, prime_sol::Function, aux_sol::Function; max_iter=1000, tol=1e-3)

    m = mpc_para.m
    n = mpc_para.n
    ρ = mpc_para.rho
    cost_func = mpc_para.cost_func
    prime_sol = prime_sol
    aux_sol   = aux_sol

    function solve(data, para; verbose=false)

        # Initialize scaled variables
        X = Matrix{FloatType}(I,n,n)      
        Z = copy(X)
        U = zeros(n,n)                  

        for k in 1:max_iter

            # ---------------- X-step (Y update) ----------------
            q = Z - U
            X, ME = prime_sol(q)  # solve logdet prox
    
            # Save training data if needed
            push!(data["org_f"], cost_func(X))
            push!(data["input"], triangle_vec(q))
            push!(data["env"],   ME)
            push!(data["grad"],  ρ*triangle_vec(q - X))

            # ---------------- Z-step (projection) ---------------
            Z = aux_sol(X + U, para)

            # ---------------- Dual update -----------------------
            U += (X - Z)

            # ---------------- Convergence check -----------------
            if maximum(abs.(X - Z)) < tol
                if verbose
                    println("ADMM converged at iteration $k")
                end
                break
            end

        end

        return Z, cost_func(Z)
    end

    return solve
end


function prime_solver_data(name::String, mpc_para::data_opt)
    n  = mpc_para.n
    ρ  = mpc_para.rho

    model = pick_solver(name)
    set_silent(model)

    @variable(model, Y[1:n, 1:n], PSD)

    # Log-det cone variable
    @variable(model, t)

    # Parametric input matrix Q
    @variable(model, q[1:n, 1:n] in MOI.Parameter.(zeros(n, n)))

    @constraint(model, [t; 1.; triangle_vec(Y)] in MOI.LogDetConeTriangle(n))

    @objective(model, Min, -t + 0.5 * ρ * sum((Y[i,j] - q[i,j])^2 for j in 1:n for i in j:n)) #upper

    optimize!(model)

    
    return function(q_in::Matrix{FloatType})
        # Set Q parameter
        MOI.set.(model, POI.ParameterValue(), model[:q], q_in)

        optimize!(model)

        return value.(model[:Y]), objective_value(model)
    end
end


function aux_solver_data(name::String, mpc_para::data_opt)
    model = pick_solver(name)
    set_silent(model)

    m = mpc_para.m
    n = mpc_para.n
    A = mpc_para.A
    
    # Z is PSD matrix
    @variable(model, Z[1:n, 1:n], PSD)

    @variable(model, q[1:n, 1:n] in MOI.Parameter.(zeros(n, n)))

    @objective(model, Min, sum((Z[i,j] - q[i,j])^2 for j in 1:n for i in j:n))

    @constraint(model, c[i in 1:m], dot(A[:, i], Z, A[:, i]) <= 1)


    # Pre-solve / warm-start build
    optimize!(model)

    return @inbounds function (q_in::Matrix{FloatType}, para::Matrix{FloatType})
        MOI.set.(model, POI.ParameterValue(), model[:q], (q_in))
        # --- Constraints: a_i' * X * a_i ≤ 1 for i = 1..m ---
        delete(model, c)
        unregister(model, :c)

        @constraint(model, c[i in 1:m], dot(para[:, i], Z, para[:, i]) <= 1)

        optimize!(model)

        return value.(model[:Z])
    end
end
