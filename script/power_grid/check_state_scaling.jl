# Run with julia --project=. --threads=8 script/power_grid/check_state_scaling.jl
# Uses existing inputs; does not overwrite benchmark results.
using Test
include("power_grid_table.jl")

function check_state_scaling()
    instances = npzread(INPUT_FILE)
    load = vec(instances["load"][1, :])
    gen = vec(instances["gen"][1, :])
    x0s = sort(vec(instances["x0"]))
    x0s = x0s[round.(Int, range(1, length(x0s); length=20))]
    target = GapTarget()
    reference = nlp_runner("IPOPT", target; reference=true)
    refs = [get_objective(mpc_data, reference(x0, load, gen).vars) for x0 in x0s]

    scales = isempty(ARGS) ? (1.0, 10.0, 30.0, 100.0, 300.0, 2000.0) : parse.(Float64, ARGS)
    for scale in scales
        proj = dynamics_projection(mpc_data; state_scale=scale)
        q = zeros(Float64, 4, N)
        v = Matrix(proj(q, 0.5, load, gen))
        @test maximum(abs, proj(v, 0.5, load, gen) - v) < 1e-8
        q[4, 1] = 1.0
        @test maximum(abs, proj(q, 0.5, load, gen) - v) > 1e-10
        @test maximum(abs, mpc_data.A .* vcat(0.5, v[4, 1:end-1]) .+
                          mpc_data.B .* v[2, :] .- v[4, :]) < 1e-8
        @test abs(v[4, end] - 0.5) < 1e-8
        @test maximum(abs, v[1, :] + v[2, :] + gen - load - v[3, :]) < 1e-8

        solve = LME_ADMM_split(mpc_data, gradient_struct(model, s_mb, dim), proj)
        iter = Ref(0)
        residual = Ref(Inf)
        jref = Ref(refs[1])
        cb = function (z, w, a, v, b, i, J, t)
            iter[] = i
            residual[] = max(maximum(abs, w - v), maximum(abs, v - z))
            return gap_percent(J, jref[]) <= G_OPT
        end
        solve(x0s[1], load, gen, cb; tol=tol, max_iter=MAX_ITER) # Warm-up.
        times, iterations, gaps = Float64[], Int[], Float64[]
        converged = 0
        for (x0, Jref) in zip(x0s, refs)
            jref[] = Jref
            v, seconds = solve(x0, load, gen, cb; tol=tol, max_iter=MAX_ITER)
            gap = gap_percent(get_objective(mpc_data, v), Jref)
            converged += residual[] < tol && gap <= G_OPT &&
                         eco_solution_feasible(mpc_data, v, x0, load, gen, tol)
            push!(times, 1000seconds)
            push!(iterations, iter[])
            push!(gaps, gap)
        end
        println((state_scale=scale, converged=converged, samples=length(x0s),
                 mean_iterations=mean(iterations), max_iterations=maximum(iterations),
                 mean_ms=mean(times), max_gap_pct=maximum(gaps)))
    end
end

check_state_scaling()
