# Pass an existing IPOPT-reference NPZ from power_grid_table.jl as ARGS[1].
# Checks the actual runner on every instance without overwriting benchmark files.
using Test
include("power_grid_table.jl")

function check_smel_performance(path)
    data = npzread(path)
    inputs = npzread(INPUT_FILE)
    for key in ("N", "seed", "x0", "load", "gen")
        @test data[key] == inputs[key]
    end
    Random.seed!(SEED)
    q = randn(dim, N)
    g = gradient_struct(model, N, dim)
    grad = copy(g(q))
    function model_value(x)
        z = NNlib.softplus.(model.U0 * x + model.b0)
        for layer in model.layers
            z = NNlib.softplus.(layer.W * z + layer.U * x + layer.b)
        end
        return dot(model.v, z) + dot(model.a, x) + model.c
    end
    for j in (1, N), i in 1:dim
        h = 1e-5
        qp, qm = copy(q), copy(q)
        qp[i,j] += h
        qm[i,j] -= h
        numerical = (model_value(qp[:,j]) - model_value(qm[:,j])) / (2h)
        @test grad[i,j] ≈ numerical atol=1e-7 rtol=1e-6
    end
    vals = reshape([-1000., -100., -1., 0., 1., 100., 1000.], :, 1)
    activ, sigma = similar(vals), similar(vals)
    activation_sigma!(activ, sigma, vals)
    @test activ ≈ NNlib.softplus.(vals)
    @test sigma ≈ NNlib.σ.(vals)
    activation_sigma_only!(sigma, vals, activ)
    @test sigma ≈ NNlib.σ.(vals)

    target = GapTarget()
    target.percent = G_OPT
    run = admm_runner("sMEL-ADMM", target)
    load = vec(data["load"][1,:])
    gen = vec(data["gen"][1,:])
    target.reference = data["J_opt"][1]
    run(data["x0"][1], load, gen) # Warm-up, excluded.
    times, walls, gaps, violations = Float64[], Float64[], Float64[], Float64[]
    iterations = Int[]
    for k in eachindex(data["x0"])
        target.reference = data["J_opt"][k]
        x0 = data["x0"][k]
        elapsed = @elapsed r = run(x0, load, gen)
        @test r.status == "converged"
        @test r.stopping_residual < tol
        @test eco_solution_feasible(mpc_data, r.vars, x0, load, gen, tol)
        gap = gap_percent(get_objective(mpc_data, r.vars), target.reference)
        @test gap <= G_OPT
        push!(times, 1000r.seconds)
        push!(walls, 1000elapsed)
        push!(iterations, r.iterations)
        push!(gaps, gap)
        push!(violations, feasibility(r.vars, x0, load, gen))
        k % GC_EVERY == 0 && GC.gc()
    end
    println((samples=length(times), tol=tol, state_scale=SMEL_STATE_SCALE,
             mean_ms=mean(times), max_ms=maximum(times),
             mean_wall_ms=mean(walls), max_wall_ms=maximum(walls),
             mean_iterations=mean(iterations), max_gap_pct=maximum(gaps),
             max_constraint_violation=maximum(violations)))
end

check_smel_performance(only(ARGS))
