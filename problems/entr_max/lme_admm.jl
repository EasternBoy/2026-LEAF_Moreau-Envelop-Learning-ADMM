using LDLFactorizations
include(joinpath(@__DIR__, "..", "..", "src", "kkt.jl"))     # kkt_matrix, AffineProjection (the v-step)

function aux_solver_gen(solver_name::String, para_opt::data_opt)
    model = pick_solver(solver_name)

    n = para_opt.n
    scale = var_scale(para_opt)
    @variable(model,   x[1:n]      .>= 1e-9)
    @variable(model,   query[1:n]  in MOI.Parameter.(zeros(n)))
    @constraint(model, sum(x) == scale)

    @constraint(model, para_opt.A*x .<= para_opt.b*scale)

    J = dot(x, x) - 2*dot(query, x)
    @objective(model, Min, J)

    set_silent(model)
    optimize!(model) #build model

    return function solver(q::Vector{FloatType})
        MOI.set.(model, POI.ParameterValue(), model[:query], q)
        optimize!(model)

        return JuMP.value.(model[:x]), JuMP.solve_time(model)
    end
end


function LME_ADMM(data::data_opt, gradient::gradient_struct, aux_sol::Function)

    n      = data.n
    scale  = var_scale(data)
    z      = zeros(FloatType, n)
    w      = zeros(FloatType, n)
    α      = zeros(FloatType, n)
    buffer = zeros(FloatType, n)

    n_mb = div(n-1, vector_chunk(gradient)) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)


    let ρ   = data.rho
        return @inbounds function solver(data::data_opt, callback::Union{Function, Nothing} =  nothing; 
            tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false, γ = 1.6)

            fill!(w, 0.)
            fill!(α, 0.)
            fill!(buffer, 0.)

            total_time = 0
            J = 0.

            for i in 1:max_iter
                # ==== z-update ====
                start_time = time_ns()
                @. buffer = w + α
                z .= buffer .- mini_batch(local_gradients, buffer)./ρ

                @. z = γ * z + (1 - γ) * w

                total_time += time_ns() - start_time

                # ==== w-update ====
                w, sol_time = aux_sol(z .- α)
                total_time += sol_time*1e9

                ## ============== Calculate dual variables and check termination ===========
                start_time = time_ns()
                @. buffer  = w - z
                @. α += buffer

                CALL_BACK_STATUS = false
                total_time += time_ns() - start_time

                if callback !== nothing
                    J = get_objective(data, w/scale)
                    CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
                end

                residual = maximum(abs.(buffer))

                TERMINATION_STATUS = CALL_BACK_STATUS || (residual < tol)


                if TERMINATION_STATUS
                    J = get_objective(data, w/scale)
                    if verbose  println("LME-ADMM converges at iteration $i with objective value = $J and residual = $residual")  end
                    break 
                end

                if i == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.")  end
            end

            return w/scale, total_time/1e9, J
        end
    end
end



@inbounds function sLME_ADMM(data::data_opt, gradient::gradient_struct, callback::Union{Function, Nothing} =  nothing; 
    tol::FloatType = 1e-2, max_iter::Int = 1000, verbose::Bool = false)

    n = data.n
    m = data.m
    ρ = data.rho
    scale = var_scale(data)

    z       = zeros(FloatType, n+m)
    w       = copy(z)
    v       = copy(z)
    α       = copy(z)
    β       = copy(z)
    buffer1 = copy(z)
    buffer2 = copy(z)

    n_mb = div(n-1, vector_chunk(gradient)) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)

    # equality constraints on [x; s]:  A x + s = scale·b,  1ᵀx = scale
    M = [sparse(data.A) sparse(I, m, m); sparse(ones(1, n)) spzeros(1, m)]
    K = kkt_matrix(M)

    start_time    = time_ns()
    proj = AffineProjection(K, [scale .* data.b; scale])

    J = 0

    for i in 1:max_iter
        # ==== z-update ====
        @. buffer1 = v + β
        z[1:n]     .= buffer1[1:n] .- mini_batch(local_gradients, buffer1[1:n])./ρ
        z[n+1:n+m] .= buffer1[n+1:n+m]

        # ==== v-update ====
        # Use the equality constraints in v-update
        @. buffer1 = (z - β + w + α)/2
        # v .= aux_sol(buffer1)
        v[1:n+m]   .= proj(buffer1[1:n+m])
        
        # ==== w-update ====
        # Use the inequality constraints in v-update
        @. buffer2 = v - α
        @. w = max(buffer2, 1e-9)

        ## ============== Calculate dual variables and check termination ===========
        @. buffer1  = w - v
        @. buffer2  = v - z
        @. α += ρ * buffer1
        @. β += ρ * buffer2

        CALL_BACK_STATUS = true   # without a callback, stop on the residual alone

        if callback !== nothing
            J = get_objective(data, w[1:n] ./ scale)   # w = [x; s]: exclude the m slacks
            CALL_BACK_STATUS = callback(z, w, α, v, β, i, J)
        end

        residual = max(maximum(abs.(buffer1)), maximum(abs.(buffer2)))
        TERMINATION_STATUS = CALL_BACK_STATUS && (residual < tol)

        ## ============== Check termination ===========
        if TERMINATION_STATUS
            if verbose
                J = get_objective(data, w[1:n] ./ scale)
                println("sLME-ADMM converges at iteration $i with objective value = $J and residual = $residual")  
            end
            break 
        end

        if i == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.") end
    end

    solving_time = (time_ns() - start_time)/1e9

    return v ./ scale, solving_time, J
end



# @inbounds function eq_proj(para_opt::data_opt, KKT::SparseMatrixCSC)
#     # An analytic solution for  min ||[x,s] - q||² s.t.   M[x, s] = b
#     # KKT matrix 

#     n = para_opt.n
#     m = para_opt.m

#     K  = spzeros(n+2m+1, n+2m+1)
#     Im = sparse(I, m, m)

#     @views K[n+m+1:n+2m, 1:n]      = sparse(para_opt.A)
#     @views K[n+m+1:n+2m, n+1:n+m]  = Im
 
#     @views K[1:n, n+m+1:n+2m]      = para_opt.A'
#     @views K[n+1:n+m, n+m+1:n+2m]  = Im


#     @views K[n+2m+1, 1:n]  = ones(n)'
#     @views K[1:n, n+2m+1]  = ones(n)

#     @views K[1:n+m, 1:n+m] = sparse(I, n+m, n+m)
#     @time F   = ldl(K)

#     RHS = zeros(FloatType, n+2m+1)
#     RHS[n+m+1:n+2m]  .= scale * para_opt.b
#     RHS[n+2m+1]      .= scale

#     return @inbounds function solver(q::Vector{FloatType})

#         RHS[1:n+m] .= q

#         x = F \ RHS
#         return x[1:n+m]
#     end
# end


# function eq_proj(para_opt::data_opt)
#     # An analytic solution for  min ||[x,s] - q||² s.t.   Ax = s, sum(x) = 1
#     # KKT matrix 

#     n = para_opt.n
#     m = para_opt.m
#     K = spzeros(n+2m+1, n+2m+1)

#     @views K[n+m+1:n+2m, 1:n]     = para_opt.A
#     @views K[n+m+1:n+2m, n+1:n+m] = -I(m)

#     @views K[1:n, n+m+1:n+2m]     = para_opt.A'
#     @views K[n+1:n+m, n+m+1:n+2m] = -I(m)


#     @views K[n+2m+1, 1:n]  = ones(n)'
#     @views K[1:n, n+2m+1]  = ones(n)

#     @views K[1:n+m, 1:n+m] = I(n+m)

#     F   = ldl(K)

#     RHS         = zeros(FloatType, n+2m+1)
#     RHS[n+2m+1] = scale

#     return @inbounds @views function solver(q::Vector{FloatType})

#         @. RHS[1:n+m] = q

#         x = F \ RHS
#         return x[1:n+m]
#     end
# end



# function LME_ADMM_split(data::data_opt, gradient::gradient_struct)
#     """
#     ADMM with splitting the constraints into two update steps, one for dynamics and one for bounds.
#     The v-update uses the dynamics constraints, while the w-update uses the bound constraints.
#     The dynamics constraints are used for consensus: v = z and w = v.
#     """
#     n       = data.n
#     m       = data.m
#     z       = zeros(FloatType, n+m)
#     w       = copy(z)
#     v       = copy(z)
#     α       = copy(z)
#     β       = copy(z)
#     buffer1 = copy(z)
#     buffer2 = copy(z)

#     n_mb = div(n-1, s_mb) + 1
#     local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)

#     let n   = data.n,
#         m   = data.m,
#         ρ   = data.rho
#         return @inbounds function solver(data::data_opt, callback::Union{Function, Nothing} =  nothing; 
#             tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false)

#             start_time    = time_ns()

#             K  = spzeros(n+2m+1, n+2m+1)
#             Im = sparse(I, m, m)

#             K[1:n+m,      1:n+m]      = sparse(I, n+m, n+m)
#             K[n+m+1:n+2m, 1:n]        = data.A
#             K[1:n,        n+m+1:n+2m] = data.A'
#             K[n+m+1:n+2m, n+1:n+m]    = Im
#             K[n+1:n+m,    n+m+1:n+2m] = Im
#             K[n+2m+1,     1:n]        = ones(n)'
#             K[1:n,        n+2m+1]     = ones(n)


#             F   = ldl(K)
#             RHS = zeros(FloatType, n+2m+1)
#             RHS[n+m+1:n+2m]  .= scale * data.b
#             RHS[n+2m+1]       = scale


#             fill!(w, 0.)
#             fill!(α, 0.)
#             fill!(v, 0.)
#             fill!(β, 0.)

#             J = 0

#             for i in 1:max_iter
#                 # ==== z-update ====
#                 @. buffer1 = v + β
#                 z[1:n]     .= buffer1[1:n] .- mini_batch(local_gradients, buffer1[1:n])./ρ
#                 z[n+1:n+m] .= buffer1[n+1:n+m]

#                 # ==== v-update ====
#                 # Use the equality constraints in v-update
#                 @. buffer1 = (z - β + w + α)/2
#                 # v .= aux_sol(buffer1)
#                 RHS[1:n+m] .= buffer1[1:n+m]
#                 v[1:n+m]   .= (F \ RHS)[1:n+m]
                
#                 # ==== w-update ====
#                 # Use the inequality constraints in v-update
#                 @. buffer2 = v - α
#                 w .= max.(buffer2, 1e-9)

#                 ## ============== Calculate dual variables and check termination ===========
#                 @. buffer1  = w - v
#                 @. buffer2  = v - z
#                 α .+= buffer1
#                 β .+= buffer2

#                 CALL_BACK_STATUS = false

#                 if callback !== nothing
#                     J = get_objective(data, w ./ scale)
#                     CALL_BACK_STATUS = callback(z, w, α, v, β, i, J)
#                 end
                
#                 residual = max(maximum(abs.(buffer1)), maximum(abs.(buffer2)))
#                 TERMINATION_STATUS = CALL_BACK_STATUS || (residual < tol)

#                 ## ============== Check termination ===========
#                 if TERMINATION_STATUS
#                     if verbose
#                         J = get_objective(data, w ./ scale)
#                         println("sLME-ADMM converges at iteration $i with objective value = $J and residual = $residual")  
#                     end
#                     break 
#                 end

#                 if i == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.") end
#             end

#             solving_time = (time_ns() - start_time)/1e9

#             return v ./ scale, solving_time, J
#         end
#     end
# end