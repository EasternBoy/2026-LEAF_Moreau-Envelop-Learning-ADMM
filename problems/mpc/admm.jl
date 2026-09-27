using LinearAlgebra

import ParametricOptInterface as POI
import MathOptInterface       as MOI


function ADMM_mpc(para_opt::MPCData, prime_sol_name::String; max_iter = 2000, tol = 1e-3)

    ρ  = para_opt.rho
    nx = para_opt.nx
    nu = para_opt.nu
    T  = para_opt.T
    m  = nx + nu
    n  = (T + 1) * m     # total decision variables

    prime_sol_stage, prime_sol_terminal = prime_solver_mpc(prime_sol_name, para_opt)
    aux_sol = aux_solver_mpc(para_opt)

    return function solver(data, para_opt::MPCData; verbose = false)

        z = zeros(FloatType, n)
        w = zeros(FloatType, n)
        α = zeros(FloatType, n)

        for i in 1:max_iter
            q = w + α/ρ

            for k in 0:T
                idx = k*m+1 : (k+1)*m
                q_k = q[idx]
                if k < T
                    z_k, MEq = prime_sol_stage(q_k)
                else
                    z_k, MEq = prime_sol_terminal(q_k)
                end
                z[idx] .= z_k

                # -------- Taking dataset -------
                if data !== nothing
                    push!(data["input"], q_k)
                    push!(data["env"],   MEq)
                    push!(data["grad"],  ρ .* (q_k .- z_k))
                end
            end

            w = aux_sol(z - α/ρ, para_opt)

            ## ============== Dual update ==========
            @. α += ρ*(w - z)

            ## ============== Check termination ==========
            rmax = maximum(abs.(w - z))

            if rmax <= tol
                if verbose == true
                    println("MPC ADMM converges at iteration $i---tol=$tol")
                end
                break
            end

            if i == max_iter
                println("Cannot find an accurate solution, returned a close feasibility solution.")
            end
        end

        # First control action
        u1 = w[nx+1 : m]

        # Compute MPC objective value
        J = 0.0
        for t in 1:T
            xk = w[(t-1)*m+1 : (t-1)*m+nx]
            uk = w[(t-1)*m+nx+1 : t*m]
            J += dot(xk, para_opt.Q, xk) + dot(uk, para_opt.R, uk)
        end
        xN = w[T*m+1 : T*m+nx]
        J += dot(xN, para_opt.Qt, xN)

        return u1, J
    end
end


function prime_solver_mpc(name, para_opt::MPCData)
    ρ  = para_opt.rho
    nx = para_opt.nx
    nu = para_opt.nu
    Q  = para_opt.Q
    Qt = para_opt.Qt
    R  = para_opt.R
    m  = nx + nu

    # min  x'Qx + u'Ru + (ρ/2)||z - q||²
    model_s = pick_solver(name, 1e-8)
    @variable(model_s, qs[1:m] in MOI.Parameter.(zeros(m)))
    @variable(model_s, zs[1:m])
    objective = sum(zs[1:nx]' * Q * zs[1:nx] + zs[nx+1:m]' * R * zs[nx+1:m]) + (ρ/2) * sum((zs[j] - qs[j])^2 for j in 1:m)
    @objective(model_s, Min, objective)
    optimize!(model_s)

    # ── Terminal ──
    # min  x'Qt x + (ρ/2)||z - q||²   
    model_t = pick_solver(name, 1e-8)
    @variable(model_t, qt[1:m] in MOI.Parameter.(zeros(m)))
    @variable(model_t, zt[1:m])
    terminal_objective = zt[1:nx]' * Qt * zt[1:nx] + (ρ/2) * sum((zt[j] - qt[j])^2 for j in 1:m)
    @objective(model_t, Min, terminal_objective)
    optimize!(model_t)

    stage_sol = @inbounds function(q::Vector{FloatType})
        set_parameter_value.(model_s[:qs], q)
        optimize!(model_s)
        return JuMP.value.(model_s[:zs]), objective_value(model_s)
    end

    terminal_sol = @inbounds function(q::Vector{FloatType})
        set_parameter_value.(model_t[:qt], q)
        optimize!(model_t)
        return JuMP.value.(model_t[:zt]), objective_value(model_t)
    end

    return stage_sol, terminal_sol
end


function aux_solver_mpc(para_opt::MPCData)
    model = Model(() -> Clarabel.Optimizer())
    set_silent(model)

    nx    = para_opt.nx
    nu    = para_opt.nu
    T     = para_opt.T
    A     = para_opt.A
    B     = para_opt.B
    x_min = para_opt.x_min
    x_max = para_opt.x_max
    u_min = para_opt.u_min
    u_max = para_opt.u_max
    m     = nx + nu
    n     = (T + 1) * m

    @variable(model, query[1:n] in MOI.Parameter.(zeros(n)))
    @variable(model, x0_p[1:nx] in MOI.Parameter.(zeros(nx)))
    @variable(model, w[1:n])

    # Initial condition
    @constraint(model, w[1:nx] .== x0_p)

    # Dynamics
    for t in 1:T
        @constraint(model, w[t*m+1 : t*m+nx] .== A * w[(t-1)*m+1 : (t-1)*m+nx] + B * w[(t-1)*m+nx+1 : t*m])
    end

    # State box constraints
    for t in 1:T
        @constraint(model, x_min .<= w[t*m+1 : t*m+nx] .<= x_max)
    end

    # Input box constraints
    for t in 1:T
        @constraint(model, u_min .<= w[(t-1)*m+nx+1 : t*m] .<= u_max)
    end

    # Terminal input fixed to zero
    @constraint(model, w[T*m+nx+1 : n] .== 0)

    # Projection: min ||w - q||²
    @objective(model, Min, sum((w[j] - query[j])^2 for j in 1:n))

    optimize!(model)  # build model

    return function solver(q::Vector{FloatType}, para_opt::MPCData)
        set_parameter_value.(model[:query], q)
        set_parameter_value.(model[:x0_p], para_opt.x0)
        optimize!(model)
        return JuMP.value.(model[:w])
    end
end
