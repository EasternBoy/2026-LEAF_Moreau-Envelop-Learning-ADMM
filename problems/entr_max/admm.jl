using BlockArrays
using StaticArrays


# mutable struct prime_sol_struct
#     model::Model
#     rho::FloatType
#     dim::Int
#     N::Int
# end

# function prime_sol_struct(name::String, mpc_para::data_optpy)
#     model     = pick_solver(name)
#     rho       = mpc_para.rho
#     dim       = mpc_para.dim
#     N         = mpc_para.N

#     @variable(model, m[1:N])
#     @variable(model, u[1:N])
#     @variable(model, p[1:N] .>= 0)

#     @variable(model, para[1:dim, 1:N] in MOI.Parameter.(ones(dim,N)))
#     vars = vcat(m',u',p')

#     J = sum(mpc_para.cost_func(m[i], u[i], p[i], model) + (rho/2)*(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i])) for i in 1:N)

#     @objective(model, Min, J)

#     optimize!(model) #build model

#     return prime_sol_struct(model, rho, dim, N)
# end

# function (obj::prime_sol_struct)(q::Matrix{FloatType})
#     MOI.set.(obj.model, POI.ParameterValue(), obj.model[:para], q)
#     optimize!(obj.model)

#     vars       = copy(q)
#     vars[1,:] .= JuMP.value.(obj.model[:m]) 
#     vars[2,:] .= JuMP.value.(obj.model[:u]) 
#     vars[3,:] .= JuMP.value.(obj.model[:p])
#     return vars, solve_time(obj.model)
# end


# ==========================================
# High performance version for data collection
# ==========================================
# function ADMM_eco_iter(data::data_optpy, prime_sol::prime_sol_struct, aux_sol::Function; max_iter = scale, tol = tol)

#     dim = mpc_data.dim
#     N   = mpc_data.N
#     z   = zeros(FloatType, dim, N)
#     w   = zeros(FloatType, dim, N)
#     α   = zeros(FloatType, dim, N)
#     buffer = zeros(FloatType, dim, N)

#     let N       = data.N, 
#         x0      = data.x0, 
#         load_fc = data.load_forecast[1:N], 
#         gen_fc  = data.gen_forecast[1:N]
#         return @inbounds function solver(init::FloatType, load_fc::Vector{FloatType}, gen_fc::Vector{FloatType}, callback = nothing; verbose = false)        

#             fill!(w, 0.)
#             fill!(α, 0.)

#             total_time = 0.
#             J = 0.
#             for i in 1:max_iter
#                 ## ============== Solve prime variables ===========
#                 @. buffer = w + α

#                 z, sol_time = prime_sol(buffer)
#                 total_time += sol_time
#                 ## ============== Solve auxiliary variables (QP problem) ===========
#                 w[1,:], w[2,:], w[3,:], sol_time = aux_sol(z-α, init, load_fc, gen_fc)
#                 total_time += sol_time
                
#                 ## ============== Calculate dual variables and check termination ===========
#                 start_time = time()
#                 @. buffer  = w - z
#                 @. α += buffer

#                 CALL_BACK_STATUS = false
#                 J = get_objective(data, w)
#                 total_time += time() - start_time

#                 if callback !== nothing
#                     CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
#                 end

#                 TERMINATION_STATUS = CALL_BACK_STATUS || (maximum(buffer) < tol)

#                 if TERMINATION_STATUS
#                     if verbose
#                         println("Learning ADMM converges at iteration $i with objective value = $J")
#                     end
#                     break 
#                 end

#                 if i == max_iter println("Can not find an accurate solution, returned a close feasibility solution.") end
#             end

#             return w, total_time, J
#         end
#     end
# end



# function prime_solver_eco(name::String, mpc_para::MPCData_eco)
#     model     = pick_solver(name)
#     ρ         = mpc_para.rho
#     dim       = mpc_para.dim
#     N         = mpc_para.N
#     cost_func = mpc_para.cost_func

#     @variable(model, m)
#     @variable(model, u)
#     @variable(model, p >= 0)

#     @variable(model, para[1:dim] in MOI.Parameter.(ones(3)))

#     vars = vcat(m, u, p) # should be the same order as in aux_sol

#     J = cost_func(m, u, p, model) + (ρ/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
#     @objective(model, Min, J)
#     optimize!(model) #build model

#     let model = model
#         @inbounds function solver(q::Vector{FloatType})
#             MOI.set.(model, POI.ParameterValue(), model[:para], q)
#             optimize!(model)
#             vars    = copy(q)
#             vars[1] = JuMP.value.(model[:m]) 
#             vars[2] = JuMP.value.(model[:u]) 
#             vars[3] = JuMP.value.(model[:p])
#             return vars, solve_time(model)
#         end

#         return solver
#     end
# end

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

        for i in 1:max_iter
            ## ============== Solve prime variables ===========
            for k in 1:n
                q = w[k] + α[k]
                z[k], MEq = prime_sol(q)

                # -------- Taking dataset -------
                push!(data["input"], q) #push data to global variables
                push!(data["env"],   MEq)
                push!(data["grad"],  ρ*(q - z[k]))
            end

            ## ============== Solve auxiliary variables (QP problem) ===========
            w = aux_sol(z-α, para_opt)

            ## ============== Calculate dual variables ===========
            @. α += w - z  
            
            ## ============== Check termination ===========
            rmax = maximum(abs.(w-z))

            println(rmax)

            # println(rmax)
            if rmax < tol
                if verbose == true
                    println("Conventional ADMM converges at iteration $i---tol=$tol")
                end
                break 
            end

            if i == max_iter
                println("Can not find an accurate solution, returned a close feasibility solution.")
            end
        end


        J_ADMM = sum(para_opt.cost_func.(w/scale))

        return w/scale, J_ADMM
    end
end






function prime_solver_eco_data(name, para_opt)
    model = pick_solver(name, 1e-8)

    ρ         = para_opt.rho
    cost_func = para_opt.cost_func
    scale     = var_scale(para_opt)

    @variable(model, query in MOI.Parameter(0))
    @variable(model, x >= 1e-9)

    J = cost_func(x) - x*log(scale) + (ρ/2)*(x - query)^2
    @objective(model, Min, J)
    optimize!(model) #build model

    return @inbounds function solver(q::FloatType)
        MOI.set.(model, POI.ParameterValue(), model[:query], q)
        optimize!(model)
        return JuMP.value.(model[:x]), objective_value(model)
    end
end


function aux_solver_eco_data(para_opt)
    model = Model(() -> Gurobi.Optimizer())
    set_silent(model)

    n = para_opt.n
    m = para_opt.m
    scale = var_scale(para_opt)

    @variable(model,   x[1:n] .>= 1e-8)
    @variable(model,   query[1:n]  in  MOI.Parameter.(zeros(n)))
    @constraint(model, sum(x) == scale)

    @variable(model, A[1:m, 1:n] in MOI.Parameter.(para_opt.A))
    @variable(model, b[1:m]      in MOI.Parameter.(para_opt.b*scale))
    @constraint(model, A * x .<= b)


    J = dot(x, x) - 2*dot(query, x)
    @objective(model, Min, J)

    optimize!(model) #build model

    return function solver(q::Vector{FloatType}, para_opt::data_opt)
        MOI.set.(model, POI.ParameterValue(), model[:A], para_opt.A)
        MOI.set.(model, POI.ParameterValue(), model[:b], para_opt.b*scale)
        MOI.set.(model, POI.ParameterValue(), model[:query], q)

        optimize!(model)
        
        return JuMP.value.(model[:x])
    end
end




function prox_operator(ρ::FloatType, scale)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model,   x >= 1e-9)
    @variable(model,   query  in  MOI.Parameter(0))


    J = x*(log(x/scale)) + (ρ/2)*(x - query)^2
    @objective(model, Min, J)

    return @inbounds function solver(q::FloatType)
        MOI.set.(model, POI.ParameterValue(), model[:query], q)
        optimize!(model)
        return JuMP.value.(model[:x]), objective_value(model)
    end
end


function data_gen!(range_q, data_collect::Dict)

    data = data_opt()
    prox = prox_operator(data.rho, var_scale(data))

    @simd for q in range_q
        prox_val, ME = prox(q)
        push!(data_collect["input"], q)
        push!(data_collect["env"],   ME)
        push!(data_collect["grad"],  data.rho * (q - prox_val))
    end

end