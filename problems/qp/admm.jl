using BlockArrays
using StaticArrays
import ParametricOptInterface as POI
import MathOptInterface as MOI

# ==========================================
# Less efficient version for data collection
# ==========================================
function ADMM_eco_iter_data(para_opt::data_opt, prime_sol_name::String; max_iter = 100, tol = 1e-3)

    ρ  = para_opt.rho
    n  = para_opt.n
    scale = var_scale(para_opt)

    prime_sol   = prime_solver_eco_data(prime_sol_name, para_opt)
    aux_sol     = aux_solver_eco_data(para_opt)


    return function solver(data, para_opt::data_opt; verbose = false)

        z = zeros(FloatType, n)
        w = zeros(FloatType, n)
        α = zeros(FloatType, n)
        samples = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])
        converged = false
        primal_residual = dual_residual = Inf

        for i in 1:max_iter
            ## ============== Solve prime variables ===========
            q = w + α
            z, MEq = prime_sol(q)

            # -------- Taking dataset -------
            push!(samples["input"], q)
            push!(samples["env"],   MEq)
            push!(samples["grad"],  ρ*(q - z))

            ## ============== Solve auxiliary variables (QP problem) ===========
            w_previous = copy(w)
            w = aux_sol(z-α, para_opt)

            ## ============== Calculate dual variables ===========
            @. α += w - z

            ## ============== Check termination ===========
            primal_residual = maximum(abs, w - z)
            dual_residual = ρ * maximum(abs, w - w_previous)

            if verbose
                @printf("ADMM %4d: primal=%.3e, dual=%.3e\n", i, primal_residual, dual_residual)
            end
            if max(primal_residual, dual_residual) < tol
                converged = true
                break
            end
        end

        converged || error("ADMM did not converge in $max_iter iterations: primal=$primal_residual, dual=$dual_residual; samples were not accepted")
        for key in ("input", "env", "grad")
            append!(data[key], samples[key])
        end


        J_ADMM = get_objective(para_opt, w/scale)

        return w/scale, J_ADMM
    end
end






function prime_solver_eco_data(name, para_opt)
    @assert lowercase(name) == "osqp"
    model = Model(() -> POI.Optimizer(MOI.instantiate(OSQP.Optimizer; with_cache_type = Float64)))
    set_optimizer_attribute(model, "eps_abs", 1e-10)
    set_optimizer_attribute(model, "eps_rel", 1e-10)
    set_silent(model)

    ρ = para_opt.rho
    n = para_opt.n

    @variable(model, query[1:n] in MOI.Parameter.(zeros(n)))
    @variable(model, x[1:n])

    J = 0.5*dot(x, para_opt.Q*x) + dot(para_opt.p, x) + (ρ/2)*sum((x - query).^2)
    @objective(model, Min, J)
    optimize!(model) #build model

    return @inbounds function solver(q::Vector{FloatType})
        set_parameter_value.(model[:query], q)
        optimize!(model)
        is_solved_and_feasible(model) || error("OSQP prox failed: $(termination_status(model))")
        return JuMP.value.(model[:x]), objective_value(model)
    end
end


function aux_solver_eco_data(para_opt)
    model = Model(() -> POI.Optimizer(MOI.instantiate(OSQP.Optimizer; with_cache_type = Float64)))
    set_optimizer_attribute(model, "eps_abs", 1e-10)
    set_optimizer_attribute(model, "eps_rel", 1e-10)
    set_silent(model)

    n = para_opt.n
    scale = var_scale(para_opt)

    @variable(model, x[1:n])
    @variable(model, query[1:n] in MOI.Parameter.(zeros(n)))
    @variable(model, rhs[1:para_opt.neq] in MOI.Parameter.(para_opt.x*scale))
    @constraint(model, para_opt.A*x .== rhs)
    @constraint(model, para_opt.G*x .<= para_opt.h*scale)

    J = dot(x, x) - 2*dot(query, x)
    @objective(model, Min, J)
    optimize!(model) #build model

    return function solver(q::Vector{FloatType}, para_opt::data_opt)
        set_parameter_value.(model[:rhs], para_opt.x*scale)
        set_parameter_value.(model[:query], q)
        optimize!(model)
        is_solved_and_feasible(model) || error("OSQP projection failed: $(termination_status(model))")
        return JuMP.value.(model[:x])
    end
end


function prox_operator(para_opt::data_opt)
    return prime_solver_eco_data("osqp", para_opt)
end


function data_gen!(range_q, data_collect::Dict)
    data = data_opt()
    prox = prox_operator(data)

    for q in range_q
        prox_val, ME = prox(q)
        push!(data_collect["input"], q)
        push!(data_collect["env"],   ME)
        push!(data_collect["grad"],  data.rho * (q - prox_val))
    end
end
