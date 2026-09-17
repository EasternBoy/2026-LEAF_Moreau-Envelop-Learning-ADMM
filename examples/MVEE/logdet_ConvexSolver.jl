using Convex
using MathOptInterface, MosekTools
const MOI = MathOptInterface

function mpc_logdet_solver(name::String, data, init)
    # For now we just enforce that you are using MOSEK
    @assert lowercase(name) == "mosek" "Only 'mosek' is supported in this implementation."

    m   = data.m
    n   = data.n
    eps = data.eps

    @assert size(init) == (n, m) "init must be an n×m matrix (columns are a_i)."

    model = Model(MosekTools.Optimizer)
    set_silent(model)  # remove this if you want solver log

    # --- Decision variable: X ∈ Sⁿ, PSD ---
    @variable(model, X[1:n, 1:n], PSD)

    # --- Enforce X ≽ eps * I (diagonal ≥ eps) ---
    for i in 1:n
        @constraint(model, X[i, i] >= eps)
    end

    # --- Constraints: a_i' * X * a_i ≤ 1 for i = 1..m ---
    for i in 1:m
        ai = init
        push!(constraints, ai[:, i]'*X*ai[:, i] <= 1)
    end

    problem = minimize(cost_func(X), constraints)
    problem.constraints = vcat(problem.constraints, X >= eps * I(n)) # ensure X is positive definite
    

    solve!(problem, solver, silent = false)

    return evaluate(X), problem.optval
end