# ADMM baselines for QP + L1, both with the exact prox of f = ½y'Qy + p'y + λ‖y‖₁ (FISTA, prox_env in
# problem.jl):
#   ADMM_exact     split ADMM: the sMEL-ADMM splitting of problems/qp/lme_admm.jl (variables [y; s/D],
#                  affine projection, nonnegative slacks) with the exact prox in place of the learned
#                  envelope; same stopping test and callback as sLME_ADMM
#   sLME_ADMM_corrected  sLME-ADMM with the learned prox followed by exact ISTA correction steps
#   ADMM_standard  standard ADMM, as problems/qp/admm.jl: z = prox_f(w + α), w = projection of z - α
#                  onto {y : Ay = x, Gy ≤ h} (a QP, solved by OSQP), α += w - z
import ParametricOptInterface as POI
import MathOptInterface as MOI

@inbounds function ADMM_exact(data::data_opt, callback::Union{Function, Nothing} = nothing;
    tol::FloatType = 1e-4, feas_tol::FloatType = 1e-6,
    max_iter::Int = 1000, projection = qp_projection(data))

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
    x_obj   = zeros(FloatType, n)

    start_time = time_ns()
    proj = projection.proj
    @views proj.RHS[n+m+1:n+m+data.neq] .= scale .* data.x
    @views proj.RHS[n+m+data.neq+1:end] .= scale .* data.h ./ projection.slack_scale

    for i in 1:max_iter
        # ==== z-update: exact prox of f/ρ on the y block ====
        @. buffer1 = v + β
        z[1:n] .= first(prox_env(buffer1[1:n]; ρ = ρ, λ = data.λ))   # buffer1[1:n] copies: prox_env takes a Vector
        @views z[n+1:n+m] .= buffer1[n+1:n+m]

        # ==== v-update: projection onto {M u = b} ====
        @. buffer1 = (z - β + w + α)/2
        project!(v, proj, buffer1)

        # ==== w-update: slacks nonnegative, y free ====
        @. buffer2 = v - α
        @views w[1:n] .= buffer2[1:n]
        @views @. w[n+1:n+m] = max(buffer2[n+1:n+m], 0.0)

        # ==== dual updates and termination ====
        @. buffer1 = w - v
        @. buffer2 = v - z
        @. α += buffer1
        @. β += buffer2

        residual = max(maximum(abs, buffer1), maximum(abs, buffer2))
        @views @. x_obj = v[1:n] / scale
        feasible = qp_solution_feasible(data, x_obj, feas_tol)
        status = callback === nothing ? true :
            callback(z, w, α, v, β, i, get_objective(data, x_obj), (time_ns() - start_time)/1e9)
        status && residual < tol && feasible && break
        i == max_iter && println("ADMM reached $max_iter iterations without convergence: residual=$residual; returning the last iterate.")
    end

    solving_time = (time_ns() - start_time)/1e9
    y = v[1:n] ./ scale
    return y, solving_time, get_objective(data, y)
end


"""
The projection onto {y : Ay = x, Gy ≤ h} of the standard ADMM, min ‖y - u‖² by OSQP (tolerance 1e-10,
as problems/qp/admm.jl).  Built once; `proj(u, data)` updates the parameters u and x and solves.
"""
function feasible_projection(data::data_opt; tol = 1e-10)
    model = Model(() -> POI.Optimizer(MOI.instantiate(OSQP.Optimizer; with_cache_type = Float64)))
    set_optimizer_attribute(model, "eps_abs", tol)
    set_optimizer_attribute(model, "eps_rel", tol)
    set_optimizer_attribute(model, "max_iter", 100_000)
    set_silent(model)
    n = data.n
    @variable(model, y[1:n])
    @variable(model, u[1:n] in MOI.Parameter.(zeros(n)))
    @variable(model, rhs[1:data.neq] in MOI.Parameter.(data.x))
    @constraint(model, data.A * y .== rhs)
    @constraint(model, data.G * y .<= data.h)
    @objective(model, Min, dot(y, y) - 2dot(u, y))
    optimize!(model)
    return function proj(uval::Vector{FloatType}, d::data_opt)
        set_parameter_value.(model[:rhs], d.x)
        set_parameter_value.(model[:u], uval)
        optimize!(model)
        is_solved_and_feasible(model) || error("OSQP projection failed: $(termination_status(model))")
        return JuMP.value.(model[:y])
    end
end

"Standard ADMM for QP + L1 (see the file header); returns y = w (feasible), the solving time and J(y)."
function ADMM_standard(data::data_opt, proj, callback::Union{Function, Nothing} = nothing;
    tol::FloatType = 1e-4, feas_tol::FloatType = 1e-6, max_iter::Int = 1000)

    n, ρ = data.n, data.rho
    z = zeros(FloatType, n); w = zeros(FloatType, n); α = zeros(FloatType, n)
    start_time = time_ns()
    for i in 1:max_iter
        z = first(prox_env(w + α; ρ = ρ, λ = data.λ))   # prox of f/ρ
        w = proj(z - α, data)                          # projection onto the feasible set
        α .+= w - z
        residual = maximum(abs, w - z)
        feasible = qp_solution_feasible(data, w, feas_tol)
        status = callback === nothing ? true :
            callback(z, w, α, nothing, nothing, i, get_objective(data, w), (time_ns() - start_time)/1e9)
        status && residual < tol && feasible && break
        i == max_iter && println("standard ADMM reached $max_iter iterations without convergence: residual=$residual; returning the last iterate.")
    end
    solving_time = (time_ns() - start_time)/1e9
    return w, solving_time, get_objective(data, w)
end

# sLME-ADMM with correction steps: the learned prox x₀ = q − ∇ICNN(q)/ρ, then `steps` exact
# proximal-gradient (ISTA) steps on the prox subproblem ½y'(Q + ρI)y + (p − ρq)'y + λ‖y‖₁,
#     x ← S_{λ/L}(x − ((Q + ρI)x + p − ρq)/L),   L = λ_max(Q) + ρ,
# each contracting the prox error by 1 − (λ_min(Q) + ρ)/L.  Otherwise as sLME_ADMM (problems/qp/lme_admm.jl).
@inbounds function sLME_ADMM_corrected(data::data_opt, gradient::gradient_struct, steps::Int,
    callback::Union{Function, Nothing} = nothing;
    tol::FloatType = 1e-4, feas_tol::FloatType = 1e-6, max_iter::Int = 1000, projection = qp_projection(data))

    n = data.n
    m = data.nineq
    ρ = data.rho
    Lc = QP_L1_EIGS[2] + ρ
    τ = data.λ / Lc

    z       = zeros(FloatType, n+m)
    w       = copy(z)
    v       = copy(z)
    α       = copy(z)
    β       = copy(z)
    buffer1 = copy(z)
    buffer2 = copy(z)
    grad    = zeros(FloatType, n)
    q       = zeros(FloatType, n)
    x       = zeros(FloatType, n)
    g       = zeros(FloatType, n)
    x_obj   = zeros(FloatType, n)

    n_mb = div(n-1, vector_chunk(gradient)) + 1
    local_gradients = ntuple(_ -> deepcopy(gradient), n_mb + 1)

    start_time = time_ns()
    proj = projection.proj
    @views proj.RHS[n+m+1:n+m+data.neq] .= data.x
    @views proj.RHS[n+m+data.neq+1:end] .= data.h ./ projection.slack_scale

    for i in 1:max_iter
        # ==== z-update: learned prox + correction steps ====
        @. buffer1 = v + β
        @views q .= buffer1[1:n]
        mini_batch!(grad, local_gradients, q)
        @. x = q - grad/ρ
        for _ in 1:steps
            mul!(g, data.Q, x)
            @. g += data.p + ρ * (x - q)
            @. x = soft_threshold(x - g/Lc, τ)
        end
        @views z[1:n] .= x
        @views z[n+1:n+m] .= buffer1[n+1:n+m]

        # ==== v-update ====
        @. buffer1 = (z - β + w + α)/2
        project!(v, proj, buffer1)

        # ==== w-update ====
        @. buffer2 = v - α
        @views w[1:n] .= buffer2[1:n]
        @views @. w[n+1:n+m] = max(buffer2[n+1:n+m], 0.0)

        # ==== duals and termination ====
        @. buffer1 = w - v
        @. buffer2 = v - z
        @. α += buffer1
        @. β += buffer2
        residual = max(maximum(abs, buffer1), maximum(abs, buffer2))
        @views x_obj .= v[1:n]
        feasible = qp_solution_feasible(data, x_obj, feas_tol)
        status = callback === nothing ? true :
            callback(z, w, α, v, β, i, get_objective(data, x_obj), (time_ns() - start_time)/1e9)
        status && residual < tol && feasible && break
        i == max_iter && println("sLME-ADMM (corrected) reached $max_iter iterations without convergence: residual=$residual; returning the last iterate.")
    end

    solving_time = (time_ns() - start_time)/1e9
    y = v[1:n]
    return y, solving_time, get_objective(data, y)
end
