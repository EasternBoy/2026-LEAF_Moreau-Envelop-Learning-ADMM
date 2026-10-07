# Save a QP row summary, following experiments/entr_max/row.jl.
# julia --project=. experiments/qp/row.jl [100 50 50] [--force]
include(joinpath(@__DIR__, "table.jl")) # Helpers only; does not run the benchmark.
const C_V = 1e-4

function run_row()
    need = ["ground_truth-$(SUFFIX).npz", "OSQP-$(SUFFIX)-tol=$(OSQP_TOL).npz",
            "sLME-ADMM-$(SUFFIX)-tol=$(SLME_TOL)-gopt=$(max_opt_gap).npz"]
    ("--force" in ARGS || !all(file -> isfile(joinpath(OUT, file)), need)) && main()
end

fmt_time(x) = @sprintf("%.3f (%.3f)", mean(x), maximum(x))
fmt_gap(x) = all(isfinite, x) ? @sprintf("%.3g (%.3g)", mean(x), maximum(x)) : "undefined (non-finite solution)"
fmt_viol(x) = @sprintf("%.1e (%.1e)", mean(x), maximum(x))

function time_cell(saved)
    saved === nothing && return "—", Inf
    ok = all(saved["max_viol"] .<= C_V) && all(saved["gap_pct"] .<= max_opt_gap)
    return ok ? (fmt_time(saved["time_ms"]), mean(saved["time_ms"])) : ("unable to achieve", Inf)
end

function render_row()
    files = ["OSQP-$(SUFFIX)-tol=$(OSQP_TOL).npz",
             "sLME-ADMM-$(SUFFIX)-tol=$(SLME_TOL)-gopt=$(max_opt_gap).npz",
             "DC3-$(SUFFIX).npz"]
    X = instances()
    Jopt = npzread(joinpath(OUT, "ground_truth-$(SUFFIX).npz"))["J_opt"]
    results = [isfile(joinpath(OUT, file)) ? npzread(joinpath(OUT, file)) : nothing for file in files]
    for saved in results
        saved === nothing && continue
        merge!(saved, score(X, saved["W"], Jopt))
    end
    cells = time_cell.(results)
    best = argmin(last.(cells))
    timing = [isfinite(cell[2]) && i == best ? "**$(cell[1])**" : cell[1] for (i, cell) in enumerate(cells)]
    violation = [saved === nothing ? "—" : fmt_viol(saved["max_viol"]) for saved in results]
    gap = [saved === nothing ? "—" : fmt_gap(saved["gap_pct"]) for saved in results]
    rows = ["| Solving time (g_opt ≤ $(max_opt_gap)%) | $n | $neq | $m | $(join(timing, " | ")) |",
            "| Constr. viol. | $n | $neq | $m | $(join(violation, " | ")) |",
            "| Opt. gap (%) | $n | $neq | $m | $(join(gap, " | ")) |"]
    md = """
    # QP: n = $n, neq = $neq, m = $m

    Solving time in ms and optimality gap in %, as mean (maximum) over
    $(N_SAMPLES) instances. Constraint violation is the maximum original-coordinate
    equality/inequality violation per instance, also shown as mean (maximum).
    **Bold** marks the lowest mean time among methods whose every saved solution
    meets the $(max_opt_gap)% objective-gap target and feasibility threshold c_v = $(C_V).
    "unable to achieve" means a target was missed; — means results are missing.
    Timings retain the boundaries described in README.md; this script does not retrain DC3.

    |  | n | neq | m | OSQP mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
    |---|---|---|---|---|---|---|
    $(join(rows, "\n"))
    """
    mkpath(OUT)
    write(joinpath(OUT, "qp_row-$(SUFFIX)-tol=$(SLME_TOL)-gopt=$(max_opt_gap).md"), md)
    println(md)
end

if abspath(PROGRAM_FILE) == @__FILE__
    dimensions = filter(arg -> !startswith(arg, "--"), ARGS)
    isempty(dimensions) || (length(dimensions) == 3 && parse.(Int, dimensions) == [n, neq, m]) ||
        error("usage: julia --project=. experiments/qp/row.jl [100 50 50] [--force]")
    run_row()
    render_row()
end
