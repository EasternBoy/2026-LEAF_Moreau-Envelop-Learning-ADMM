using BlockArrays
using StaticArrays


mutable struct prime_sol_struct
    model::Model
    rho::FloatType
    dim::Int
    N::Int
end

function prime_sol_struct(name::String, mpc_para::MPCData_eco)
    model     = pick_solver(name)
    rho       = mpc_para.rho
    dim       = mpc_para.dim
    N         = mpc_para.N

    @variable(model, m[1:N])
    @variable(model, u[1:N])
    @variable(model, p[1:N] .>= 0)

    @variable(model, para[1:dim, 1:N] in MOI.Parameter.(ones(dim,N)))
    vars = vcat(m',u',p')

    J = sum(mpc_para.cost_func(m[i], u[i], p[i], model) + (rho/2)*(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i])) for i in 1:N)

    @objective(model, Min, J)

    optimize!(model) #build model

    return prime_sol_struct(model, rho, dim, N)
end

function (obj::prime_sol_struct)(q::Matrix{FloatType})
    set_parameter_value.(obj.model[:para], q)
    optimize!(obj.model)

    vars       = copy(q)
    vars[1,:] .= JuMP.value.(obj.model[:m]) 
    vars[2,:] .= JuMP.value.(obj.model[:u]) 
    vars[3,:] .= JuMP.value.(obj.model[:p])
    return vars, solve_time(obj.model)
end


# ==========================================
# High performance version for data collection
# ==========================================
function ADMM_eco_iter(data::MPCData_eco, prime_sol::prime_sol_struct, aux_sol::Function; max_iter = 1000, tol = tol)

    dim = mpc_data.dim
    N   = mpc_data.N
    z   = zeros(FloatType, dim, N)
    z₊  = zeros(FloatType, dim, N)

    w   = zeros(FloatType, dim, N)
    α   = zeros(FloatType, dim, N)
    buffer = zeros(FloatType, dim, N)

    let N       = data.N, 
        x0      = data.x0, 
        load_fc = data.load_forecast[1:N], 
        gen_fc  = data.gen_forecast[1:N]
        return @inbounds function solver(init::FloatType, load_fc::Vector{FloatType}, gen_fc::Vector{FloatType}, callback = nothing; verbose = false, γ = 1.2)        

            fill!(w, 0.)
            fill!(α, 0.)

            total_time = 0.
            J = 0.
            for i in 1:max_iter
                ## ============== Solve prime variables ===========
                @. buffer = w + α

                z, sol_time = prime_sol(buffer)

                @. z = γ * z + (1 - γ) * w

                total_time += sol_time
                ## ============== Solve auxiliary variables (QP problem) ===========
                w[1,:], w[2,:], w[3,:], sol_time = aux_sol(z - α, init, load_fc, gen_fc)
                total_time += sol_time
                
                ## ============== Calculate dual variables and check termination ===========
                start_time = time()
                @. buffer  = w - z
                @. α += buffer

                CALL_BACK_STATUS = false
                J = get_objective(data, w)
                total_time += time() - start_time

                if callback !== nothing
                    CALL_BACK_STATUS = callback(z, w, α, i, J, total_time)
                end

                TERMINATION_STATUS = CALL_BACK_STATUS || (maximum(buffer) < tol)

                if TERMINATION_STATUS
                    if verbose
                        println("Conventional ADMM converges at iteration $i with objective value = $J")
                    end
                    break 
                end

                if i == max_iter println("Can not find an accurate solution, returned a close feasibility solution.") end
            end

            return w, total_time, J
        end
    end
end



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

function aux_solver_eco(solver_name::String, mpc_para::MPCData_eco)
    model = pick_solver(solver_name)

    N    = mpc_para.N
    dim  = mpc_para.dim
    BESS = mpc_para.BESS
    dT   = mpc_para.dT

    u_max = mpc_para.u_max
    u_min = mpc_para.u_min

    x_max   = mpc_para.x_max
    x_min   = mpc_para.x_min
    x0_init = mpc_para.x0

    load_fc = mpc_para.load_forecast
    gen_fc  = mpc_para.gen_forecast

    @variable(model, m[1:N])
    @variable(model, u_min  .<= u[1:N] .<= u_max)
    @variable(model, p[1:N] .>= 1)
    @variable(model, x_min  .<= x[0:N] .<= x_max)

    @variable(model, para[1:dim,1:N] in MOI.Parameter.(ones(dim,N)))
    @variable(model, x0             in MOI.Parameter.(x0_init))
    @variable(model, load[1:N]      in MOI.Parameter.(load_fc[1:N]))
    @variable(model, generator[1:N] in MOI.Parameter.(gen_fc[1:N]))

    for i in 0:N-1 #Dynamics
        @constraint(model, x[i+1] == x[i] -  dT*u[i+1]/BESS)
    end

    @constraint(model, x[N] >= mpc_para.x_end_min)    #End constraint
    @constraint(model, x[0] == x0)    #Initial state

    @constraint(model, u + m + generator - load - p .== 0) #Power flow

    vars = vcat(m', u', p')
    J    = sum(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i]) for i in 1:N)
    @objective(model, Min, J)

    optimize!(model) #build model

    let model = model
        return function solver(q::Matrix{FloatType}, init::FloatType, load::Vector{FloatType}, generator::Vector{FloatType})
            set_parameter_value.(model[:para], q)
            set_parameter_value.(model[:load], load)
            set_parameter_value.(model[:generator], generator)
            set_parameter_value(model[:x0], init)

            optimize!(model)
            
            return JuMP.value.(model[:m]), JuMP.value.(model[:u]), JuMP.value.(model[:p]), solve_time(model)
        end
    end    
end



# ==========================================
# Less efficient version for data collection
# ==========================================
function ADMM_eco_iter_data(mpc_para::MPCData_eco, prime_sol::Function, aux_sol::Function; max_iter = 1000, tol = tol)

    N   = mpc_para.N
    dim = mpc_para.dim
    ρ   = mpc_para.rho

    function solver(data, init = mpc_para.x0, load_fc = mpc_para.load_forecast[1:N], gen_fc = mpc_para.gen_forecast[1:N]; verbose = false)

        z = rand(dim, N)
        w = rand(dim, N)
        α = rand(dim, N)

        for i in 1:max_iter
            ## ============== Solve prime variables ===========
            for k in 1:N
                q = w[:,k] + α[:,k]
                z[:,k], MEq = prime_sol(q)

                # -------- Taking dataset -------
                push!(data["input"], q) #push data to global variables
                push!(data["env"],   MEq)
                push!(data["grad"],  ρ*(q - z[:,k]))
            end

            ## ============== Solve auxiliary variables (QP problem) ===========
            αᵣ = 1.1
            z₊ = αᵣ*z + (1 - αᵣ)*w
            w = aux_sol(z₊ - α, init, load_fc, gen_fc)

            ## ============== Calculate dual variables ===========
            # α += w - z  
            α += w - z₊
            
            ## ============== Check termination ===========
            rmax = maximum(abs.(w-z))

            # println(rmax)
            if rmax < tol
                if verbose == true
                    println("Conventional ADMM converges at iteration $i---tol=$tol")
                end
                break 
            end
        end

        J_ADMM = sum(cost_func(w[1,k], w[2,k], w[3,k]) for k in 1:N)

        return w, J_ADMM
    end

    return solver
end


function prime_solver_eco_data(name::String, mpc_para::MPCData_eco)
    model = pick_solver(name)
    ρ     = mpc_para.rho
    N     = mpc_para.N
    cost_func = mpc_para.cost_func

    @variable(model, m)
    @variable(model, u)
    @variable(model, p >= 0)

    @variable(model, para[1:3] in MOI.Parameter.(ones(3)))

    vars = [m, u, p] # should be the same order as in aux_sol

    J = cost_func(m, u, p, model) + (ρ/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
    @objective(model, Min, J)
    optimize!(model) #build model

    let model = model
        return @inbounds function solver(q::Vector{FloatType})
            set_parameter_value.(model[:para], q)
            optimize!(model)
            vars = [JuMP.value.(model[:m]), JuMP.value.(model[:u]), JuMP.value.(model[:p])]
            return vars, objective_value(model)
        end
    end

    return solver
end


function aux_solver_eco_data(solver_name::String, mpc_para::MPCData_eco)
    model = pick_solver(solver_name)

    N    = mpc_para.N
    BESS = mpc_para.BESS
    dT   = mpc_para.dT

    u_max = mpc_para.u_max
    u_min = mpc_para.u_min

    x_max   = mpc_para.x_max
    x_min   = mpc_para.x_min
    x0_init = mpc_para.x0

    load_fc = mpc_para.load_forecast
    gen_fc  = mpc_para.gen_forecast

    @variable(model, m[1:N])
    @variable(model, u_min  .<= u[1:N] .<= u_max)
    @variable(model, p[1:N] .>= 0)
    @variable(model, x_min  .<= x[0:N] .<= x_max)

    @variable(model, para[1:3, 1:N] in MOI.Parameter.(ones(3, N)))
    @variable(model, x0             in MOI.Parameter.(x0_init))
    @variable(model, load[1:N]      in MOI.Parameter.(load_fc[1:N]))
    @variable(model, generator[1:N] in MOI.Parameter.(gen_fc[1:N]))

    for i in 0:N-1 #Dynamics
        @constraint(model, x[i+1] == x[i] -  dT*u[i+1]/BESS)
    end

    @constraint(model, x[N] >= mpc_para.x_end_min)    #End constraint
    @constraint(model, x[0] == x0)    #Initial state

    @constraint(model, u + m + generator - load - p .== 0) #Power flow

    vars = vcat(m', u', p')
    J = sum(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i]) for i in 1:N)
    @objective(model, Min, J)

    optimize!(model) #build model

    function solver(q::Matrix{FloatType}, init::FloatType, load::Vector{FloatType}, generator::Vector{FloatType})
        set_parameter_value.(model[:para], q)
        set_parameter_value.(model[:load], load)
        set_parameter_value.(model[:generator], generator)
        set_parameter_value(model[:x0], init)

        optimize!(model)
        
        vars = vcat(JuMP.value.(model[:m])',  JuMP.value.(model[:u])', JuMP.value.(model[:p])')

        return vars
    end

    return solver
end
