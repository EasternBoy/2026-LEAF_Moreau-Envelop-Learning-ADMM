"""
OSQP on the slack (epigraph) form of the QP + L1 problem, a QP in (y, t):
    min ½y'Qy + p'y + λ𝟏't  s.t.  Ay = x, Gy ≤ h, -t ≤ y ≤ t.
Returns y, the solver's internal time, and the objective in y.
"""
function JuMP_solver(name, para_opt, tol; verbose::Bool = false)
    scale = var_scale(para_opt)
    model = pick_solver(name, tol)
    if name == "osqp"   # the default 4000 iterations stop short of 1e-8 on this problem
        set_optimizer_attribute(model, "max_iter", 100_000)
        set_optimizer_attribute(model, "polishing", true)
    end
    n = para_opt.n

    @variable(model, y[1:n])
    @variable(model, t[1:n])

    @constraint(model, para_opt.A*y .== para_opt.x*scale)
    @constraint(model, para_opt.G*y .<= para_opt.h*scale)
    @constraint(model, y .<= t)
    @constraint(model, -t .<= y)

    @objective(model, Min, 0.5 * dot(y, para_opt.Q * y)/scale + dot(para_opt.p, y) + para_opt.λ * sum(t))

    set_silent(model)
    optimize!(model)
    is_solved_and_feasible(model) || error("$name failed: $(termination_status(model))")

    yv = JuMP.value.(y)/scale
    verbose && @printf("objective value = %5.4f, solve time = %5.3f", get_objective(para_opt, yv), JuMP.solve_time(model))
    return yv, JuMP.solve_time(model), get_objective(para_opt, yv)
end
