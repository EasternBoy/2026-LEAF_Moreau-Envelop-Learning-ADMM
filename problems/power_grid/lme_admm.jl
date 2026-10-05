# Check the returned (normalized) iterate, including the strict domain p > 0.
function eco_solution_feasible(data, v, init, load, gen, tol)
    all(isfinite, v) || return false
    m, u, p = eachrow(@view v[1:3, :])
    if size(v, 1) == 3
        x = similar(u)
        previous_state = init
        for k in eachindex(u)
            x[k] = data.A * previous_state + data.B * u[k]
            previous_state = x[k]
        end
    else
        x = @view v[4, :]
    end
    all(isfinite, x) || return false
    all(>(0), p) || return false
    previous = vcat(init, x[1:end-1])
    cm, cu, cp = pf_coef(data)
    eq = max(maximum(abs, data.A .* previous .+ data.B .* u .- x),
             maximum(abs, cm .* m .+ cu .* u .- cp .* p .- pf_rhs(data, load, gen)))
    viol = max(maximum(u .- data.u_max), maximum(data.u_min .- u),
               maximum(x .- data.x_max), maximum(data.x_min .- x),
               data.x_end_min - x[end], init - data.x_max, data.x_min - init, 0.0)
    return eq <= tol && viol <= tol
end

function LME_ADMM(data::MPCData_eco, gradient::gradient_struct, aux_sol::Function)

    z      = zeros(FloatType, data.dim, data.N)
    w      = zeros(FloatType, data.dim, data.N)
    α      = zeros(FloatType, data.dim, data.N)
    buffer = zeros(FloatType, data.dim, data.N)

    n_mb = div(data.N-1, column_chunk(gradient)) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)


    let N   = data.N,
        dim = data.dim,
        ρ   = data.rho
        return @inbounds function solver(x0::FloatType, load_fc::Vector{FloatType}, gen_fc::Vector{FloatType}, callback::Union{Function, Nothing} =  nothing; 
            tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false, γ = 1.2)

            fill!(w, 0.)
            fill!(α, 0.)

            total_time = 0.
            J = 0.

            for i in 1:max_iter
                # ==== z-update ====
                start_time = time()
                @. buffer = w + α
                z .= buffer .- mini_batch(local_gradients, buffer)./ρ
                @. z = γ * z + (1 - γ)* w

                total_time += time() - start_time

                # ==== w-update ====
                w[1,:], w[2,:], w[3,:], sol_time = aux_sol(z - α, x0, load_fc, gen_fc)
                total_time += sol_time

                ## ============== Calculate dual variables and check termination ===========
                start_time = time()
                @. buffer  = w - z
                @. α += buffer

                CALL_BACK_STATUS = false
                total_time += time() - start_time

                J = get_objective(data, w)

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
                end

                residual = maximum(abs, buffer)
                feasible = eco_solution_feasible(data, w, x0, load_fc, gen_fc, tol)
                TERMINATION_STATUS = residual < tol && feasible &&
                                     (callback === nothing || CALL_BACK_STATUS)

                if TERMINATION_STATUS
                    if verbose  println("Learning ADMM converges at iteration $i with objective value = $J")  end
                    break 
                end

                if i == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.")  end
            end

            return w, total_time, J
        end
    end
end


function LME_ADMM_split(data::MPCData_eco, gradient::gradient_struct, aux_sol::Function)
    """
    ADMM with splitting the constraints into two update steps, one for dynamics and one for bounds.
    The v-update uses the dynamics constraints, while the w-update uses the bound constraints.
    The dynamics constraints are used for consensus: v = z and w = v.
    """

    z = zeros(FloatType, data.dim+1, data.N)
    w = copy(z)
    v = copy(z)
    α = copy(z)
    β = copy(z)
    buffer1 = copy(z)
    buffer2 = copy(buffer1)

    n_mb = div(data.N - 1, column_chunk(gradient)) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb)

    let N   = data.N,
        dim = data.dim,
        ρ   = data.rho,
        u_min = data.u_min,
        u_max = data.u_max,
        x_min = data.x_min,
        x_max = data.x_max,
        x_end_min = data.x_end_min
        return @inbounds function solver(init::FloatType, load_fc::Vector{FloatType}, gen_fc::Vector{FloatType}, callback = nothing; 
            tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false,
            feas_tol::FloatType = tol,  # feasibility tolerance of the returned v
            γ::FloatType = 1.0)         # over-relaxation of the z-update (1 = none)

            for state in (z, w, v, α, β, buffer1, buffer2)
                fill!(state, 0.)
            end

            J = 0
            total_time = 0.

            for i in 1:max_iter
                start_time = time_ns()
                # ==== z-update ====
                buffer1 .= v .+ β
                @views z[1:dim, :] .= buffer1[1:dim, :] .-
                    mini_batch(local_gradients, buffer1[1:dim, :])./ρ
                @views copyto!(z[dim+1, :], buffer1[dim+1, :])  # no learning for state variable
                γ == 1 || (@. z = γ * z + (1 - γ) * v)

                # ==== v-update ====
                # Use the equality constraints in v-update
                @. buffer1 = (z - β + w + α)/2
                v .= aux_sol(buffer1, init, load_fc, gen_fc)
                
                # ==== w-update ====
                # Use the inequality constraints in v-update
                @. w = v - α

                @views clamp!(w[2,:], u_min, u_max)
                @views clamp!(w[3,:], 0.,    Inf)
                @views clamp!(w[4,:], x_min, x_max)
                w[4,N] = clamp(w[4,N], max(x_min, x_end_min), x_max)  # terminal bound

                ## ============== Calculate dual variables and check termination ===========
                @. buffer1  = w - v
                @. buffer2  = v - z
                α .+= buffer1
                β .+= buffer2

                total_time += time_ns() - start_time
                CALL_BACK_STATUS = false
                J = get_objective(data, v)

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, v, β, i, J)
                end

                residual = max(maximum(abs, buffer1), maximum(abs, buffer2))
                feasible = eco_solution_feasible(data, v, init, load_fc, gen_fc, feas_tol)
                # A callback cannot bypass consensus or returned-solution feasibility.
                TERMINATION_STATUS = residual < tol && feasible &&
                                     (callback === nothing || CALL_BACK_STATUS)

                ## ============== Check termination ===========
                if TERMINATION_STATUS
                    if verbose
                        println("Learning ADMM spliting constraints (v-update) converges at iteration $i with objective value = $J")
                    end
                    break 
                end
            end

            return v, total_time / 1e9
        end
    end
end

@inbounds function get_objective(data::MPCData_eco, z::Matrix{FloatType})
    J = sum(data.cost_func(z[1,k], z[2,k], z[3,k]) for k in 1:data.N)
    return J
end
