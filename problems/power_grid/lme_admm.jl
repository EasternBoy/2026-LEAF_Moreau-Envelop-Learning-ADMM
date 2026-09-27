# Check the returned iterate, including the strict domain p > 0.
function eco_solution_feasible(data, v, init, load, gen, tol)
    all(isfinite, v) || return false
    m, u, p, x = eachrow(v)
    all(>(0), p) || return false
    previous = vcat(init, x[1:end-1])
    eq = max(maximum(abs, data.A .* previous .+ data.B .* u .- x),
             abs(x[end] - init), maximum(abs, u .+ m .+ gen .- load .- p))
    viol = max(maximum(u .- data.u_max), maximum(data.u_min .- u),
               maximum(x .- data.x_max), maximum(data.x_min .- x),
               init - data.x_max, data.x_min - init, 0.0)
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

                TERMINATION_STATUS = CALL_BACK_STATUS || (maximum(abs, buffer) < tol)

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
        x_max = data.x_max
        return @inbounds function solver(init::FloatType, load_fc::Vector{FloatType}, gen_fc::Vector{FloatType}, callback = nothing; 
            tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false)

            for state in (z, w, v, α, β, buffer1, buffer2)
                fill!(state, 0.)
            end

            J = 0
            start_time = time()

            for i in 1:max_iter
                # ==== z-update ====
                buffer1 .= v .+ β
                @views z[1:dim, :] .= buffer1[1:dim, :] .- mini_batch(local_gradients, buffer1[1:dim, :])./ρ
                @views copyto!(z[dim+1, :], buffer1[dim+1, :])  # no learning for state variable

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

                ## ============== Calculate dual variables and check termination ===========
                @. buffer1  = w - v
                @. buffer2  = v - z
                α .+= buffer1
                β .+= buffer2

                CALL_BACK_STATUS = false
                J = get_objective(data, v)

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, v, β, i, J)
                end

                residual = max(maximum(abs, buffer1), maximum(abs, buffer2))
                feasible = eco_solution_feasible(data, v, init, load_fc, gen_fc, tol)
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

            return v, time() - start_time
        end
    end
end

@inbounds function get_objective(data::MPCData_eco, z::Matrix{FloatType})
    J = sum(data.cost_func(z[1,k], z[2,k], z[3,k]) for k in 1:data.N)
    return J
end


# function  res_J_update(data::MPCData_eco, z::Matrix{FloatType}, w::Matrix{FloatType}, α::Matrix{FloatType})
#     @inbounds @simd for i in eachindex(α)
#         α[i] += w[i] - z[i]
#     end

#     J       = get_objective(data, w)
#     opt_gap = 100abs(J - Jopt)/Jopt #w is already feasible

#     return opt_gap, J
# end

# function res_J_update(data::MPCData_eco, z::Matrix{FloatType}, w::Matrix{FloatType}, α::Matrix{FloatType}, v::Matrix{FloatType}, β::Matrix{FloatType})
#     @inbounds @simd for i in eachindex(β)
#         β[i] += v[i] - z[i]
#         α[i] += w[i] - v[i]
#     end

#     J       = get_objective(data, v)
#     opt_gap = 100abs(J - Jopt)/Jopt + 1e9norm(w .- v, Inf) #only feasible solutions are used
#     return opt_gap, J
# end

# function res_update(z::Matrix{FloatType}, w::Matrix{FloatType}, α::Matrix{FloatType})
#     res = similar(z) 
#     @inbounds @simd for i in eachindex(α)
#         r = w[i] - z[i]
#         α[i] += r
#         res[i] = r
#     end

#     return maximum(res)
# end

# function res_update(z::Matrix{FloatType}, w::Matrix{FloatType}, α::Matrix{FloatType}, v::Matrix{FloatType}, β::Matrix{FloatType})
#     max_res = copy(β) 
#     @inbounds @simd for i in eachindex(β)
#         r1 = v[i] - z[i]
#         r2 = w[i] - v[i]
#         β[i] += r1
#         α[i] += r2

#         max_res[i] = abs(r1) > abs(r2) ? abs(r1) : abs(r2) 
#     end
#     return maximum(max_res)
# end


# mutable struct sLME_ADMM

#     u_min::FloatType
#     u_max::FloatType
#     x_min::FloatType
#     x_max::FloatType
#     ρ::FloatType
#     gradient::gradient_struct
#     aux_sol::Function
#     z::MMatrix
#     w::MMatrix
#     v::MMatrix
#     α::MMatrix
#     β::MMatrix
#     buffer::MMatrix
# end