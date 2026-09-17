using ParametricOptInterface, MathOptInterface
using ParameterJuMP
using JuMP
using BlockArrays
using CSV, DataFrames

const POI = ParametricOptInterface
const MOI = MathOptInterface

function ADMM_eco_iter(mpc_para, prime_sol::Function, aux_sol::Function; max_iter = 1000, tol = tol)

    nx = mpc_para.nx
    N  = mpc_para.N
    ρ  = mpc_para.rho

    function solver(init = mpc_para.x0, load_tr = mpc_para.load_track; data = nothing, verbose = false)

        z = zeros(nx+1, N)
        w = rand(nx+1,  N)
        β = rand(nx+1,  N)

        total_time = 0.

        for i in 1:max_iter
            ## ============== Solve prime variables ===========
            time_loop = 0.
            for k in 1:N
                q = w[:,k] + β[:,k]
                z[:,k], solve_time, MEq = prime_sol(q)
                time_loop += solve_time

                # -------- Taking dataset -------
                if data !== nothing
                    push!(data["input"], q) #push data to global variables
                    push!(data["env"],  MEq)
                    push!(data["grad"],  ρ*(q - z[:,k]))
                end
            end
            total_time += time_loop/N

            ## ============== Solve auxiliary variables (QP problem) ===========
            w, solve_time = aux_sol(z-β, init, load_tr)
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
        
        J_ADMM = sum(cost_func(z[1:nx, k], z[nx+1, k]) for k in 1:N)

        return w, J_ADMM, total_time
    end

    return solver
end


function prime_solver_eco(name::String, mpc_para)
    model = pick_solver(name)
    nx = mpc_para.nx
    ρ  = mpc_para.rho

    @variable(model, vars[1:nx+1])
    set_start_value.(vars, ones(nx+1))

    @variable(model, para[1:nx+1] in MOI.Parameter.(ones(nx+1)))

    J = cost_func(vars[1:nx], vars[nx+1], model) + (ρ/2)*(dot(vars, vars) - 2*dot(para, vars) + dot(para, para))
    @objective(model, Min, J)
    optimize!(model) #build model

    function solver(q)
        MOI.set.(model, POI.ParameterValue(), model[:para], q)
        optimize!(model)
        return JuMP.value.(model[:vars]), solution_summary(model).solve_time, objective_value(model)
    end

    return solver
end



function aux_solver_economic(name::String, mpc_para)
    model = pick_solver(name)

    nx = mpc_para.nx
    N  = mpc_para.N
    u_max = mpc_para.u_max
    x_min = mpc_para.x_min
    τ      = mpc_para.tau
    track_vals = mpc_para.load_track

    @variable(model, vars[1:nx+1, 1:N])

    @variable(model, para[1:nx+1, 1:N] in MOI.Parameter.(ones(nx+1, N)))
    @variable(model, x0[1:nx] in MOI.Parameter.(zeros(nx)))
    @variable(model, track[1:N] in MOI.Parameter.(track_vals))

    
      
    J = sum(dot(vars[:,i], vars[:,i]) - 2*dot(para[:,i], vars[:,i]) + dot(para[:,i], para[:,i]) for i in 1:N)

    for i in 1:N-1 #Dynamics
        @constraint(model, vars[:, i+1] - vars[:, i] .<=  u_max)
        @constraint(model, vars[:, i+1] - vars[:, i] .>= -u_max)
    end


    for i in 1:N #sum constraint + relax
        @constraint(model, vars[1:nx, 1:N] .>= x_min)
        @constraint(model, sum(vars[1:nx, i]) - track[i] ==  vars[nx+1, i])
        @constraint(model, sum(vars[1:nx, i]) <= τ)
    end


    @objective(model, Min, J)
    optimize!(model) #build model

    function solver(q, init, track)
        MOI.set.(model, POI.ParameterValue(), model[:para], q)
        MOI.set.(model, POI.ParameterValue(), model[:track], track)
        MOI.set.(model, POI.ParameterValue(), model[:x0], init)

        optimize!(model)
        return JuMP.value.(model[:vars]), solution_summary(model).solve_time
    end

    return solver
end