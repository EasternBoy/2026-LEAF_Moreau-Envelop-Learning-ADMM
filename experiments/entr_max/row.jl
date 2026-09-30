length(ARGS) >= 2 || error("usage: julia --project=. row.jl n m [--force]")
const N_ROW = parse(Int, ARGS[1])
const M_ROW = parse(Int, ARGS[2])

include(joinpath(@__DIR__, "table.jl"))   # helpers only (not a worker, runs nothing)

function run_row(nn, mm)
    need = [gt_file(nn, mm); [res_file(me, g, nn, mm) for me in METHODS for (_, g) in GOPTS]]
    (FORCE || !isfile(need[1]) || !all(is_current, need[2:end])) || return
    cmd = `$(Base.julia_cmd()) --project=$REPO --threads=auto $(joinpath(@__DIR__, "table.jl")) worker $nn $mm $PASS_ARGS`
    FORCE && (cmd = `$cmd --force`)
    run(cmd)
end

function render_row(nn, mm)
    rows = String[]
    for gname in ("0.1", "1")
        gopt = parse(Float64, gname)
        cells = [time_cell("IPOPT", gname, nn, mm), time_cell("sLME-ADMM", gname, nn, mm), dc3_cell(nn, mm, gopt)]
        best = argmin(last.(cells))
        txt = [isfinite(c[2]) && i == best ? "**$(c[1])**" : c[1] for (i, c) in enumerate(cells)]
        txt[2] *= iter_suffix(gname, nn, mm)
        push!(rows, "| solving time (g_opt ≤ $(gname)%) | $nn | $mm | $(join(txt, " | ")) |")
    end
    for gname in ("0.1", "1")
        cells = [isfile(f) ? metric_cell(f, "max_viol", fmt_viol) : "—"
                 for f in (res_file("IPOPT", gname, nn, mm), res_file("sLME-ADMM", gname, nn, mm), dc3_file(nn, mm))]
        push!(rows, "| Constr. viol. (g_opt ≤ $(gname)%) | $nn | $mm | $(join(cells, " | ")) |")
    end
    ip = isfile(gt_file(nn, mm)) ? "0" : "—"
    sl = isfile(res_file("sLME-ADMM", "0.1", nn, mm)) ? metric_cell(res_file("sLME-ADMM", "0.1", nn, mm), "gap_pct", fmt_gap) : "—"
    dc = "—"
    if isfile(dc3_file(nn, mm))
        d = npzread(dc3_file(nn, mm))
        dc = is_current(dc3_file(nn, mm)) ? fmt_gap(d["gap_pct"]) : "rerun required"
        f = mean(d["feasible"] .> 0.5)
        f < 1 && (dc *= @sprintf(", feasible %.0f%%", 100f))
    end
    push!(rows, "| Opt. gap (%) | $nn | $mm | $ip | $sl | $dc |")

    md = """
    # Maximum-entropy cone program: n = $nn, m = $mm

    Solving time in ms, Constr. viol. and optimality gap in %, as mean (max) over
    $(N_SAMPLES) instances; **bold** is the lowest mean time in the row; `[k it.]` is
    sLME-ADMM's mean number of iterations; — means the data has not been produced yet.
    See entr_max_table.md / README.md for the definitions.

    |  | n | m | IPOPT mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
    |---|---|---|---|---|---|
    $(join(rows, "\n"))
    """
    mkpath(OUT_TAG)
    write(joinpath(OUT_TAG, "entr_max_row-n=$(nn)-m=$(mm).md"), md)
    println(md)
end

run_row(N_ROW, M_ROW)
render_row(N_ROW, M_ROW)
