include(joinpath(@__DIR__, "..", "..", "src", "kkt.jl"))     # kkt_matrix, AffineProjection (the v-step)
@inbounds function sLME_ADMM(data::data_opt, gradient::gradient_struct, callback::Union{Function, Nothing} =  nothing; 
    tol::FloatType = 1e-3, max_iter::Int = 1000, verbose::Bool = false)    
    
    n, m     = data.n, data.m
    ρ        = data.rho

    gradient = gradient

    dim = div(n*(n+1), 2)

    Z = zeros(dim + m)           
    V = copy(Z)
    W = copy(Z)

    β = copy(Z)
    α = copy(Z)
    
    buffer1 = copy(Z)
    buffer2 = copy(Z)
    mat     = zeros(n, n)


    gradient(rand(dim)) #Precomplie gradient
    K           = KKT_mat(data)

    start_time  = time_ns()
    proj        = AffineProjection(K, zeros(FloatType, m))
    J           = 0

    ρᵥ = 10ρ


    for k in 1:max_iter
        # ==== X-update (using learned gradient) ====
        @. buffer1      = W + α/ρ
        Z[1:dim]       .= buffer1[1:dim] .-  gradient(buffer1[1:dim])./ρ
        @views Z[dim+1:dim+m] .= buffer1[dim+1:dim+m]

        # ==== W-update ====
        @. buffer1 = (Z + V - α/ρ - β/ρᵥ)/2
        @views vec_to_sym!(mat, buffer1[1:dim], n)
        dec           = eigen(mat)
        mat          .= dec.vectors * Diagonal(max.(dec.values, 1e-10)) * dec.vectors' #P > 0
        W[1:dim]     .= triangle_vec(mat)
        @views @. W[dim+1:dim+m] = min(buffer1[dim+1:dim+m], 1) #s > 0



        # ==== V-update ====
        # Use the inequality constraints in v-update
        @. buffer2           = W + β/ρᵥ
        V                   .= proj(buffer2)

        ## ============== Calculate dual variables and check termination ===========
        @. buffer1  = W - Z
        @. buffer2  = W - V
        @. α += ρ  * buffer1
        @. β += ρᵥ * buffer2

        CALL_BACK_STATUS = false

        if callback !== nothing
            J = get_objective(data, mat)
            CALL_BACK_STATUS = callback(J)
        end
        
        residual = max(maximum(abs.(buffer1)), maximum(abs.(buffer2)))
        TERMINATION_STATUS = CALL_BACK_STATUS || (residual < tol)

        ## ============== Check termination ===========
        if TERMINATION_STATUS
            J = get_objective(data, mat)
            if verbose
                println("sLME-ADMM converges at iteration $k with objective value = $J and residual = $residual")  
            end
            break 
        end

        if k == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.") end
    end

    solving_time = (time_ns() - start_time)/1e9

    return mat, solving_time, J
end


@inbounds function get_objective(data::data_opt, w::VecOrMat)
    return data.cost_func(w)
end


@inline function vec_to_sym!(X, v, n)
    idx = 1
    @inbounds for i in 1:n
        for j in 1:i
            val     = v[idx]
            X[i, j] = val
            X[j, i] = val
            idx    += 1
        end
    end
    return idx
end

function KKT_mat(data::data_opt)
    n   = data.n
    m   = data.m
    A   = data.A
    dim = div(n*(n+1), 2)

    H = spzeros(m,      dim)

    for c in 1:m
        H[c, :] = [i==j ? A[i,c]^2 : 2A[i,c]*A[j,c] for j in 1:n for i in 1:j]
    end

    return kkt_matrix([H -sparse(I, m, m)])     # constraints H P − s = 0 on [P; s]
end



# @inbounds function sLME_ADMM2(data::data_opt, gradient::gradient_struct, callback::Union{Function, Nothing} =  nothing; 
#     tol::FloatType = 1e-3, max_iter::Int = 100, verbose::Bool = false)    
    
    
#     n, m     = data.n, data.m
#     ρ        = data.rho
#     A        = data.A
#     ρ        = data.rho

#     gradient = gradient

#     dim = div(n*(n+1), 2)

#     Z             = ones(dim + m)           
#     W             = copy(Z)
#     V             = copy(Z)

#     α             = copy(Z)
#     β             = copy(Z)
    
#     buffer1       = copy(Z)
#     buffer2       = copy(Z)

#     mat = zeros(n, n)


#     K   = KKT_mat(data)
#     RHS = zeros(FloatType, dim+2m)
#     gradient(rand(dim))

#     start_time  = time_ns()
#     F   = ldl(K)
#     J   = 0


#     for k in 1:max_iter
#         # ==== X-update (using learned gradient) ====
#         @. buffer1 = V + β
#         Z[1:dim]       .= buffer1[1:dim] .-  gradient(buffer1[1:dim])./ρ
#         Z[dim+1:dim+m] .= buffer1[dim+1:dim+m]

#         # ==== v-update ====
#         # Use the equality constraints in v-update
#         @. buffer1 = (Z - β + W + α)/2

#         RHS[1:dim+m] .= buffer1[1:dim+m]
#         V            .= (F \ RHS)[1:dim+m]

#         # ==== w-update ====
#         # Use the inequality constraints in v-update
#         @. buffer2 = V - α

#         vec_to_sym!(mat, buffer2[1:dim], n)

#         dec             = eigen(mat)

#         mat            .= dec.vectors * Diagonal(max.(dec.values, 1e-9)) * dec.vectors' #P > 0
#         W[1:dim]       .= triangle_vec(mat)
#         @. W[dim+1:dim+m] = min(buffer2[dim+1:dim+m], 1) #s > 0

#         ## ============== Calculate dual variables and check termination ===========
#         @. buffer1  = W - V
#         @. buffer2  = V - Z
#         @. α += ρ * buffer1
#         @. β += ρ * buffer2


#         CALL_BACK_STATUS = false

#         if callback !== nothing
#             J = get_objective(data, mat)
#             CALL_BACK_STATUS = callback(Z, W, α, v, β, i, J)
#         end
        
#         residual = max(maximum(abs.(buffer1)), maximum(abs.(buffer2)))
#         TERMINATION_STATUS = CALL_BACK_STATUS || (residual < tol)

#         # println("sLME-ADMM at iteration $k with residual = $residual")  


#         ## ============== Check termination ===========
#         if TERMINATION_STATUS
#             if verbose
#                 J = get_objective(data, mat)
#                 println("sLME-ADMM converges at iteration $k with objective value = $J and residual = $residual")  
#             end
#             break 
#         end

#         if k == max_iter  println("Can not find an accurate solution, returned a close feasibility solution.") end
#     end

#     solving_time = (time_ns() - start_time)/1e9

#     return mat, solving_time, J
# end