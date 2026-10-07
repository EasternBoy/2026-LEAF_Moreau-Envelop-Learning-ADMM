# Score DC3 with the same functions as OSQP/sLME-ADMM and render the combined table.
# Run from the repository root: julia --project=. experiments/qp/evaluate.jl
include("table.jl")

function evaluate()
    X = instances()
    Jopt = npzread(joinpath(OUT, "ground_truth-$(SUFFIX).npz"))["J_opt"]
    paths = ["OSQP-$(SUFFIX)-tol=$(OSQP_TOL).npz",
             "sLME-ADMM-$(SUFFIX)-tol=$(SLME_TOL)-gopt=$(max_opt_gap).npz",
             "DC3-$(SUFFIX).npz"]
    results = Pair{String, Any}[]
    for (name, filename) in zip(["Optimizer (OSQP)", "sLME-ADMM", "DC3 + correction"], paths)
        path = joinpath(OUT, filename)
        saved = npzread(path)
        @assert size(saved["W"]) == (N_SAMPLES, n)
        @assert length(saved["time_ms"]) == N_SAMPLES
        merge!(saved, score(X, saved["W"], Jopt))
        npzwrite(path, saved)
        push!(results, name => saved)
    end
    render(results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluate()
end
