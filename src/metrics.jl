# Scoring shared by every method in a comparison table (IPOPT, sLME-ADMM, DC3 + correction).
# Solvers only save their raw solutions; every reported optimality gap, constraint
# violation and feasibility flag is computed by the functions in this file.

const FEAS_TOL = 1e-4        # a solution is feasible when its largest constraint violation is ≤ FEAS_TOL
const METRICS_VERSION = 4    # stored in every scored result file; bump when a rule below changes

"Relative optimality gap in % of `J` against the reference optimum `J_ref`."
opt_gap(J, J_ref) = 100abs(J - J_ref) / abs(J_ref)

"Per-instance gap, violation and feasibility from objective values and violations."
function score(J, viol, J_ref)
    gap = opt_gap.(J, J_ref)
    return (gap_pct = gap, max_viol = viol, feasible = isfinite.(gap) .& (viol .<= FEAS_TOL))
end

# ---------------------------------------------------------------------------
# Maximum-entropy problem:  min Σ wᵢ log wᵢ  s.t.  1ᵀw = 1,  A w ≤ b
# The objective is scored at max(w, 0), with 0 log 0 = 0; any −w is counted in
# the violation instead, so a slightly negative entry does not make the gap undefined.

entr_max_objective(w) = all(isfinite, w) ? sum(x -> x <= 0 ? 0.0 : x * log(x), w) : NaN
entr_max_violation(A, b, w) = max(maximum(A * w .- b), maximum(-w), abs(sum(w) - 1), 0.0)

"""
    score_entr_max(A, b, W, J_ref)

`A` (N, m, n) and `b` (N, m) are the instances, `W` (N, n) the solutions of one
method and `J_ref` (N,) the reference optima.  Returns `(gap_pct, max_viol, feasible)`.
"""
function score_entr_max(A, b, W, J_ref)
    N = size(W, 1)
    J    = [entr_max_objective(W[k, :]) for k in 1:N]
    viol = [entr_max_violation(A[k, :, :], b[k, :], W[k, :]) for k in 1:N]
    return score(J, viol, J_ref)
end
