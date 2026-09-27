using ParametricOptInterface, MathOptInterface
using ParameterJuMP
using JuMP

const POI = ParametricOptInterface
const MOI = MathOptInterface

function ADMM_iter_cstr_learn(mpc_para::MPCData, prime_model::Function, aux_model::Function; max_iter = 1000, tol = tol)
    
    nx = mpc_para.nx
    nu = mpc_para.nu
    N  = mpc_para.N
    nz = nx + nu
    ρ = mpc_para.rho

    
    function solver(init = mpc_para.x0; data = nothing, verbose = false)
        z = zeros(nz, N+1)
        w = rand(nz, N+1)
        β = rand(nz, N+1)

        total_time = 0.


        for i in 1:max_iter
            time_loop = 0.
            ## ============== Solve prime variables ===========
            for k in 1:N+1
                q = w[:,k] + β[:,k]
                z[:,k], solve_time, MEq = prime_model(q)
                time_loop += solve_time

                # -------- Taking dataset -------
                if data !== nothing
                    push!(data["input"], q) #push data to global variables
                    push!(data["env"],  MEq)
                    push!(data["grad"],  ρ*(q - z[:,k]))
                end
            end
            total_time += time_loop/(N+1)

            ## ============== Solve auxiliary variables (QP problem) ===========
            w, solve_time = aux_model(z - β, init)
            total_time += solve_time

            ## ============== Calculate dual variables ===========
            β += w - z  

            ## ============== Check termination ===========
            rmax = maximum(abs.(w-z))
            if rmax < tol
                if verbose == true
                    println("Conventional ADMM converges at iteration $i---Solve time=",round(total_time*1000,digits = 3),"ms","---tol=",tol)
                end
                break 
            end
        end

        J = sum(cost_func(w[1:nx,k], w[nx+1:nz,k]) for k in 1:N+1)

        return w[nx+1:nz,1:N], J, total_time
    end

    return solver
end


function prime_solver_bound_cstr(name, mpc_para)
    model = pick_solver(name)

    nx = mpc_para.nx
    nu = mpc_para.nu
    nz = nx+nu
    ρ = mpc_para.rho

    x_max = mpc_para.x_max
    x_min = mpc_para.x_min
    u_min = mpc_para.u_min
    u_max = mpc_para.u_max

    @variable(model, para[1:nz] in MOI.Parameter.(zeros(nz)))
    @variable(model, vars[1:nz])


    @constraint(model, x_min .<= vars[1:nx]       .<= x_max)
    @constraint(model, u_min .<= vars[(nx+1):nz]  .<= u_max)

    J = cost_func(vars[1:nx], vars[(nx+1):nz], model) + (ρ/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
    @objective(model, Min, J)
    optimize!(model) #build model


    function solver(q)
        MOI.set.(model, POI.ParameterValue(), model[:para], q)
        optimize!(model)
        return JuMP.value.(model[:vars]), solution_summary(model).solve_time, objective_value(model)
    end

    return solver
end



# function aux_solver(name::String)
#     model = pick_solver(name)
#     A  = dict["A"]
#     B  = dict["B"]
#     nx = dict["nx"]
#     nu = dict["nu"]
#     N  = dict["N"]

#     M₁ = [I(nx) zeros(nx, nu)]
#     M₂ = [A B]
#     nz = nx + nu

#     # Parameters
#     @variable(model, para[1:nz, 1:N+1] in MOI.Parameter.(zeros(nz, N+1)))
#     @variable(model, x0[1:nx] in MOI.Parameter.(zeros(nx)))

#     #Variables
#     @variable(model, vars[1:nz, 1:N+1])

#     @constraint(model, vars[1:nx,1] == x0)
#     for i in 1:N
#         @constraint(model, M₁*vars[:,i+1] - M₂*vars[:,i] .== 0)
#     end

#     J = 0
#     for i in 1:N+1
#         # p = z - β
#         J += dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i]) + dot(para[:,i], para[:,i])
#     end
#     @objective(model, Min, J)
#     optimize!(model) #build model

#     function solver(q, x0 = zeros(nx))
#         MOI.set.(model, POI.ParameterValue(), model[:para], q)
#         MOI.set.(model, POI.ParameterValue(), model[:x0], x0)
#         optimize!(model)
#         return JuMP.value.(model[:vars]), solution_summary(model).solve_time
#     end

#     return solver
# end

