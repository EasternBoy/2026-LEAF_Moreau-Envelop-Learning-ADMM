include("utils.jl")
using LMEADMM   # src/LMEADMM.jl


rho, mp = load_model("models/mvee/logdet-rho=3.0-m=50_ICNN.json")

model = ICNN(
    mp.U[1], 
    mp.b[1],
    [ICNN_Layer(mp.U[i], mp.W[i], mp.b[i]) for i in 2:length(mp.U)],
    mp.v, 
    mp.a,
    mp.c)

mpc_data  = data_opt()
n   = mpc_data.n
# dim = mpc_data.dim




function sLME_ADMM_callback( J::FloatType)
    opt_gap = 100abs(J - Jopt)/Jopt 

    return opt_gap < max_opt_gap
end


# NLP scaling off; mvee defines no Ipopt callbacks (it solves with Clarabel and Mosek)
pick_solver(name, tol::FloatType = 1e-6, cbs::Union{Nothing, callback_struct} = nothing) =
    solver_model(name, tol; solvers = @__MODULE__, scaling = false, madnlp_print_level = 5)
  