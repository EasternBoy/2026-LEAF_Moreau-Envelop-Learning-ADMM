include("utils.jl")
using LMEADMM   # src/LMEADMM.jl

pick_solver(name, tol::FloatType = 1e-6) =
    solver_model(name, tol; solvers = @__MODULE__)
