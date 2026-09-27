# Scores the saved solutions `W` of every method in data/cone_result with src/metrics.jl
# and writes gap_pct, max_viol and feasible back into each result file.
#
#   julia --project=. script/entropy_max/entr_max_evaluate.jl          # every (n, m)
#   julia --project=. script/entropy_max/entr_max_evaluate.jl n m      # one (n, m)
#
# entr_max_table.jl scores IPOPT and sLME-ADMM with the same functions when it solves;
# entr_max_table.py calls this script for DC3 + correction, which saves only `W`.
using NPZ, Printf, Statistics

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "data", "cone_result")
include(joinpath(REPO, "src", "metrics.jl"))

const SIZES = [(100, 1), (100, 10), (1000, 10), (1000, 100)]
const RESULT_PREFIXES = ["IPOPT-gopt=1", "IPOPT-gopt=0.1", "sLME-ADMM-gopt=1", "sLME-ADMM-gopt=0.1", "DC3"]

function evaluate(n, m)
    inst = npzread(joinpath(OUT, "instances", "instances-n=$(n)-m=$(m).npz"))
    J_ref = npzread(joinpath(OUT, "ground_truth-n=$(n)-m=$(m).npz"))["J_opt"]
    for prefix in RESULT_PREFIXES
        path = joinpath(OUT, "$(prefix)-n=$(n)-m=$(m).npz")
        isfile(path) || continue
        d = npzread(path)
        haskey(d, "W") || (@printf("n=%d m=%d %-18s no saved solutions, skipped\n", n, m, prefix); continue)
        s = score_entr_max(inst["A"], inst["b"], d["W"], J_ref)
        merge!(d, Dict("gap_pct" => s.gap_pct, "max_viol" => s.max_viol, "feasible" => s.feasible,
                       "metrics_version" => METRICS_VERSION))
        npzwrite(path, d)
        @printf("n=%d m=%d %-18s gap %.3g (%.3g) %%, viol %.1e (%.1e), feasible %.3f\n", n, m, prefix,
                mean(s.gap_pct), maximum(s.gap_pct), mean(s.max_viol), maximum(s.max_viol), mean(s.feasible))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    sizes = length(ARGS) >= 2 ? [(parse(Int, ARGS[1]), parse(Int, ARGS[2]))] : SIZES
    for (n, m) in sizes
        evaluate(n, m)
    end
end
