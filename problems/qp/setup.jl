include("utils.jl")
using LMEADMM   # src/LMEADMM.jl

function sLME_ADMM_callback(z, w, α, v, β, iter, J, elapsed = 0.0)
    opt_gap = 100abs(J - J_opt)/max(abs(J_opt), eps(FloatType))
    return opt_gap <= max_opt_gap
end

pick_solver(name, tol::FloatType = 1e-6) =
    solver_model(name, tol; solvers = @__MODULE__)
