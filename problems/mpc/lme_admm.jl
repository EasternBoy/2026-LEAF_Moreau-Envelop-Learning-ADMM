using LDLFactorizations
using Base.Threads


# Auxiliary solver for LME_ADMM_mpc: same projection as aux_solver_mpc
# but also returns JuMP solve time for total_time tracking.
function aux_solver_mpc_lme(para_opt::MPCData)
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

    @constraint(model, w[1:nx] .== x0_p)

    for t in 1:T
        @constraint(model, w[t*m+1 : t*m+nx] .== A * w[(t-1)*m+1 : (t-1)*m+nx] + B * w[(t-1)*m+nx+1 : t*m])
    end

    for t in 1:T
        @constraint(model, x_min .<= w[t*m+1 : t*m+nx] .<= x_max)
    end

    for t in 1:T
        @constraint(model, u_min .<= w[(t-1)*m+nx+1 : t*m] .<= u_max)
    end

    @constraint(model, w[T*m+nx+1 : n] .== 0)

    @objective(model, Min, dot(w, w) - 2*dot(query, w))

    optimize!(model)  # build model

    return function solver(q::Vector{FloatType}, para_opt::MPCData)
        set_parameter_value.(model[:query], q)
        set_parameter_value.(model[:x0_p], para_opt.x0)
        optimize!(model)
        return JuMP.value.(model[:w]), JuMP.solve_time(model)
    end
end


function LME_ADMM_mpc(para_opt::MPCData, gradient::gradient_struct, aux_sol::Function)

    nx = para_opt.nx
    nu = para_opt.nu
    T  = para_opt.T
    m  = nx + nu
    n  = (T + 1) * m

    z      = zeros(FloatType, n)
    w      = zeros(FloatType, n)
    α      = zeros(FloatType, n)
    buffer = zeros(FloatType, n)

    n_mb = div(n-1, s_mb) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)

    let ρ = para_opt.rho
        return @inbounds function solver(para_opt::MPCData, callback::Union{Function, Nothing} = nothing;
            tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false, γ = 1.6)

            fill!(w, 0.)
            fill!(α, 0.)
            fill!(buffer, 0.)

            total_time = 0
            J = 0.

            for i in 1:max_iter
                start_time = time_ns()
                @. buffer = w + α/ρ
                z .= buffer .- mini_batch(local_gradients, buffer)./ρ

                @. z = γ * z + (1 - γ) * w

                total_time += time_ns() - start_time

                w, sol_time = aux_sol(z .- α/ρ, para_opt)
                total_time += sol_time * 1e9

                # ── Dual update ──
                start_time = time_ns()
                @. buffer = w - z
                @. α += ρ*buffer

                CALL_BACK_STATUS = false
                total_time += time_ns() - start_time

                if callback !== nothing
                    J = compute_J_mpc(w, para_opt)
                    CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
                end

                residual = maximum(abs.(buffer))
                TERMINATION_STATUS = CALL_BACK_STATUS || (residual < tol)

                if TERMINATION_STATUS
                    J = compute_J_mpc(w, para_opt)
                    if verbose
                        println("LME-ADMM converges at iteration $i with objective value = $J and residual = $residual")
                    end
                    break
                end

                if i == max_iter
                    println("Cannot find an accurate solution, returned a close feasibility solution.")
                end
            end

            u1 = w[nx+1 : m]
            return u1, total_time/1e9, J
        end
    end
end



@inbounds function sLME_ADMM_mpc(para_opt::MPCData, gradient::gradient_struct,
    callback::Union{Function, Nothing} = nothing;
    tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false)

    nx    = para_opt.nx
    nu    = para_opt.nu
    T     = para_opt.T
    A     = para_opt.A
    B     = para_opt.B
    ρ     = para_opt.rho
    x_min = para_opt.x_min
    x_max = para_opt.x_max
    u_min = para_opt.u_min
    u_max = para_opt.u_max

    m    = nx + nu
    n    = (T + 1) * m
    n_eq = (T + 1) * nx

    M1 = [sparse(I, nx, nx) spzeros(FloatType, nx, nu)]
    M2 = [A B]

    M = spzeros(FloatType, n_eq, n)
    M[1:nx, 1:m] = M1
    for t in 1:T
        rs = t*nx + 1;  re = (t+1)*nx
        M[rs:re, t*m+1       : (t+1)*m] =  M1
        M[rs:re, (t-1)*m+1   : t*m    ] = -M2
    end

    # ── KKT matrix — factored once ──
    # Small negative regularization on dual block makes K strictly SQD for ldl
    K = [sparse(I, n, n)    M'                              ;
         M                  -1e-10*sparse(I, n_eq, n_eq)]
    F = ldl(K)

    # ── Mini-batch gradient setup ──
    n_mb            = div(n - 1, s_mb) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)

    # ── Preallocate ──
    z       = zeros(FloatType, n)
    v       = copy(z)
    w       = copy(z)
    α       = copy(z)
    β       = copy(z)
    buffer1 = copy(z)
    buffer2 = copy(z)

    g = zeros(FloatType, n_eq)
    g[1:nx] .= para_opt.x0

    RHS     = zeros(FloatType, n + n_eq)
    RHS[n+1:n+n_eq] .= g

    J = 0.
    start_time = time_ns()

    for i in 1:max_iter
        # ── z-step: learned gradient ──
        @. buffer1 = v + β/ρ
        z          .= buffer1 .- mini_batch(local_gradients, buffer1)./ρ

        # ── v-step: equality projection via KKT ──
        # Solves [I M'; M 0][v; λ] = [w + β/ρ; b]
        @. buffer1 = w + β/ρ
        RHS[1:n]        .= buffer1
        v               .= (F \ RHS)[1:n]

        # ── w-step: box projection ──
        @. buffer2 = v - α/ρ
        w[1:nx] .= buffer2[1:nx]
        for t in 1:T
            w[t*m+1 : t*m+nx] .= clamp.(buffer2[t*m+1 : t*m+nx], x_min, x_max)
        end
        for t in 1:T
            w[(t-1)*m+nx+1 : t*m] .= clamp.(buffer2[(t-1)*m+nx+1 : t*m], u_min, u_max)
        end
        w[T*m+nx+1 : n] .= 0

        # ── Dual updates ──
        @. buffer1 = w - v
        @. buffer2 = v - z
        @. α += ρ * buffer1
        @. β += ρ * buffer2

        CALL_BACK_STATUS = false
        if callback !== nothing
            J = compute_J_mpc(w, para_opt)
            CALL_BACK_STATUS = callback(z, v, w, α, β, i, J)
        end

        residual = max(maximum(abs.(buffer1)), maximum(abs.(buffer2)))
        TERMINATION_STATUS = CALL_BACK_STATUS || (residual < tol)

        if TERMINATION_STATUS
            J = compute_J_mpc(w, para_opt)
            if verbose
                println("sLME-ADMM converges at iteration $i with objective value = $J and residual = $residual")
            end
            break
        end

        if i == max_iter
            println("Cannot find an accurate solution, returned a close feasibility solution.")
        end
    end

    solving_time = (time_ns() - start_time) / 1e9
    u1 = w[nx+1 : m]
    return u1, solving_time, J
end


function compute_J_mpc(w::Vector{FloatType}, para_opt::MPCData)
    nx = para_opt.nx
    nu = para_opt.nu
    T  = para_opt.T
    m  = nx + nu
    J  = 0.0
    for t in 1:T
        xk = w[(t-1)*m+1 : (t-1)*m+nx]
        uk = w[(t-1)*m+nx+1 : t*m]
        J += dot(xk, para_opt.Q, xk) + dot(uk, para_opt.R, uk)
    end
    xN = w[T*m+1 : T*m+nx]
    J += dot(xN, para_opt.Qt, xN)
    return J
end
