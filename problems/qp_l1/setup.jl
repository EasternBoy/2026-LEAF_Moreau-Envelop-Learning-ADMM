using LMEADMM   # src/LMEADMM.jl

# sLME-ADMM is the QP solver: only the learned Moreau envelope and get_objective differ.
include(joinpath(@__DIR__, "..", "qp", "lme_admm.jl"))

# Objective-gap target in %; experiments set it before each sweep.
const GAP_TARGET = Ref(1.0)

function sLME_ADMM_callback(z, w, α, v, β, iter, J, elapsed = 0.0; J_ref = J_opt)
    opt_gap = 100abs(J - J_ref)/max(abs(J_ref), eps(FloatType))
    return opt_gap <= GAP_TARGET[]
end

pick_solver(name, tol::FloatType = 1e-6) =
    solver_model(name, tol; solvers = @__MODULE__)
