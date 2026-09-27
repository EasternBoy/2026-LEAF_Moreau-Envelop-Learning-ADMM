"""
    LMEADMM

Code shared by every problem in `problems/`: the learned Moreau envelope (ICNN and its
gradient), the sLME-ADMM v-step (affine projection through a factored KKT system), the
JuMP models of the reference solvers, and the scores reported in the comparison tables.

The repository's `Project.toml` is this package, so scripts run with `--project=.` load it
with `using LMEADMM`.  Everything is in `Float64` (`LMEADMM.FloatType`, not exported, so
scripts can keep their own `const FloatType = Float64`).
"""
module LMEADMM

using LinearAlgebra, SparseArrays, Base.Threads
using NNlib, JSON3, LDLFactorizations, JuMP
import MathOptInterface as MOI

const FloatType = Float64

include("metrics.jl")
include("icnn.jl")
include("kkt.jl")
include("solvers.jl")

# metrics.jl: optimality gap, constraint violation, feasibility
export FEAS_TOL, METRICS_VERSION, opt_gap, score, entr_max_objective, entr_max_violation, score_entr_max
# icnn.jl: the learned Moreau envelope and its gradient
export load_model, ICNN, ICNN_Layer, gradient_struct, mini_batch, vector_chunk, column_chunk,
       mul_add!, mmul_add_matrix!
# kkt.jl: the sLME-ADMM v-step
export kkt_matrix, AffineProjection
# solvers.jl: JuMP models of the reference solvers
export callback_struct, solver_model

end
