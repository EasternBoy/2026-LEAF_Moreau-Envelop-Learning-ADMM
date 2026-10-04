# Reporting uses exact objective-domain membership; no clamped objectives.
function cone_metrics(A, b, w)
    valid = all(isfinite, w) && all(>=(0), w)
    obj = valid ? sum(x -> x == 0 ? 0.0 : x * log(x), w) : NaN
    eq = abs(sum(w) - 1)
    ineq = max(maximum(A * w .- b), maximum(1e-8 / (2length(w)) .- w), 0.0)
    return (obj=obj, eq=eq, ineq=ineq, valid=valid)
end

function power_metrics(data, v, init, load, gen)
    m, u, p, x = eachrow(v)
    valid = all(isfinite, v) && all(>(0), p)
    obj = valid ? sum(data.cost_func(m[i], u[i], p[i]) for i in eachindex(p)) : NaN
    previous = vcat(init, x[1:end-1])
    eq = max(maximum(abs, data.A .* previous .+ data.B .* u .- x),
             maximum(abs, u .+ m .+ gen .- load .- p))
    ineq = max(maximum(u .- data.u_max), maximum(data.u_min .- u), maximum(-p),
               maximum(x .- data.x_max), maximum(data.x_min .- x), data.x_end_min - x[end],
               init - data.x_max, data.x_min - init, 0.0)
    return (obj=obj, eq=eq, ineq=ineq, valid=valid)
end

function metric_summary(ms, times, references; oracle=false)
    obj = [r.obj for r in ms]
    eq = [r.eq for r in ms]; ineq = [r.ineq for r in ms]
    valid = [r.valid for r in ms]
    feasible = valid .& (eq .<= 1e-4) .& (ineq .<= 1e-4)
    gaps = 100 .* abs.(obj .- references) ./ max.(abs.(references), 1e-12)
    safe_stat(f, a) = !isempty(a) && all(isfinite, a) ? f(a) : nothing
    return Dict{String,Any}(
        "obj_mean" => safe_stat(mean, obj),
        "gap_pct_mean" => safe_stat(mean, gaps), "gap_pct_max" => safe_stat(maximum, gaps),
        "gap_pct_feasible_mean" => safe_stat(mean, gaps[feasible]),
        "gap_pct_domain_valid_mean" => safe_stat(mean, gaps[valid]),
        "domain_valid_rate" => mean(valid), "constraint_feasible_rate" => mean((eq .<= 1e-4) .& (ineq .<= 1e-4)),
        "feasible_rate" => mean(feasible), "eq_max" => safe_stat(maximum, eq),
        "ineq_max" => safe_stat(maximum, ineq), "latency_median_ms" => 1e3median(times),
        "oracle_assisted" => oracle,
        "stopping_protocol" => oracle ? "oracle-assisted time to known target; reference cost excluded" : "deployment: no reference optimum used for stopping")
end
