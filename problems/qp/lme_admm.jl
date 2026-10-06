using LDLFactorizations
using LMEADMM   # src/LMEADMM.jl

@inbounds function sLME_ADMM(data::data_opt, gradient::gradient_struct,
    callback::Union{Function, Nothing} = nothing;
    tol::FloatType = 1e-4, max_iter::Int = 1000, verbose::Bool = false)

    n = data.n
    m = data.nineq
    ρ = data.rho
    scale = var_scale(data)

    z       = zeros(FloatType, n+m)
    w       = copy(z)
    v       = copy(z)
    α       = copy(z)
    β       = copy(z)
    buffer1 = copy(z)
    buffer2 = copy(z)
    grad    = zeros(FloatType, n)   # ∇ICNN at buffer1[1:n]
    x_obj   = zeros(FloatType, n)

    n_mb = div(n-1, vector_chunk(gradient)) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)

    # equality constraints on [y; s/D], D_i = ||G_i||₂.
    slack_scale = vec(sqrt.(sum(abs2, data.G; dims = 2)))
    M = [sparse(data.A) spzeros(data.neq, m); sparse(data.G ./ slack_scale) sparse(I, m, m)]
    K = kkt_matrix(M)

    start_time    = time_ns()
    proj = AffineProjection(K, scale .* [data.x; data.h ./ slack_scale])

    for i in 1:max_iter
        # ==== z-update ====
        @. buffer1 = v + β
        mini_batch!(grad, local_gradients, view(buffer1, 1:n))
        @views @. z[1:n] = buffer1[1:n] - grad/ρ
        @views z[n+1:n+m] .= buffer1[n+1:n+m]

        # ==== v-update ====
        # Use the equality constraints in v-update
        @. buffer1 = (z - β + w + α)/2
        project!(v, proj, buffer1)

        # ==== w-update ====
        # Only the inequality slacks are nonnegative; y is unrestricted.
        @. buffer2 = v - α
        @views w[1:n] .= buffer2[1:n]
        @views @. w[n+1:n+m] = max(buffer2[n+1:n+m], 0.0)

        ## ============== Calculate dual variables and check termination ===========
        @. buffer1  = w - v
        @. buffer2  = v - z
        @. α += buffer1
        @. β += buffer2

        residual = max(maximum(abs, buffer1), maximum(abs, buffer2))
        CALL_BACK_STATUS = true
        if callback !== nothing
            @views @. x_obj = v[1:n] / scale
            J = get_objective(data, x_obj)
            CALL_BACK_STATUS = callback(z, w, α, v, β, i, J, (time_ns() - start_time)/1e9)
        end
        TERMINATION_STATUS = CALL_BACK_STATUS && (residual < tol)

        ## ============== Check termination ===========
        if TERMINATION_STATUS
            if verbose
                J = get_objective(data, v[1:n] ./ scale)
                println("sLME-ADMM converges at iteration $i with objective value = $J and residual = $residual")
            end
            break
        end

        if i == max_iter println("sLME-ADMM reached $max_iter iterations without convergence: residual=$residual; returning the last iterate.") end
    end

    solving_time = (time_ns() - start_time)/1e9

    y = v[1:n] ./ scale
    return y, solving_time, get_objective(data, y)
end
