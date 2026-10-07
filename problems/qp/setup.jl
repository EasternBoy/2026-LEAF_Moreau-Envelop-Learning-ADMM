include("utils.jl")
using LMEADMM   # src/LMEADMM.jl

function sLME_ADMM_callback(z, w, α, v, β, iter, J, elapsed = 0.0; J_ref = J_opt)
    opt_gap = 100abs(J - J_ref)/max(abs(J_ref), eps(FloatType))
    return opt_gap <= max_opt_gap
end

pick_solver(name, tol::FloatType = 1e-6) =
    solver_model(name, tol; solvers = @__MODULE__)
