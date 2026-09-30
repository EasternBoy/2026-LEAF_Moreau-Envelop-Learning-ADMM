include("utils.jl")
using LMEADMM   # src/LMEADMM.jl

if !(@isdefined(GUROBI_ENV))
    GUROBI_ENV = Gurobi.Env() 
end


# A script may set POWER_GRID_MODEL before including this file to use another ICNN.
rho, mp = load_model(@isdefined(POWER_GRID_MODEL) ? POWER_GRID_MODEL : "models/power_grid/neco_mpc-rho=1.json")

model = ICNN(
    mp.U[1], 
    mp.b[1],
    [ICNN_Layer(mp.U[i], mp.W[i], mp.b[i]) for i in 2:length(mp.U)],
    mp.v, 
    mp.a,
    mp.c)

mpc_data  = energy_mag()
N   = mpc_data.N
dim = mpc_data.dim

Jopt::FloatType = 36479.1
if N == 192
    Jopt = 73117.5
end

function Ipopt_callback_BM(
   alg_mod::Cint,
   iter_count::Cint,
   obj_value::Float64,
   inf_pr::Float64,
   inf_du::Float64,
   mu::Float64,
   d_norm::Float64,
   regularization_size::Float64,
   alpha_du::Float64,
   alpha_pr::Float64,
   ls_trials::Cint,
)
    rel_opt_gap = 100abs(Jopt - obj_value)/Jopt
    stop = (rel_opt_gap < max_opt_gap) && inf_pr < 1e-4

    return !stop #False means running, True means stopping
end



function ADMM_callback(
    z::Matrix{FloatType},
    w::Matrix{FloatType},
    α::Matrix{FloatType},
    iter::Int,
    J::FloatType,
    sol_time_upto_iter::FloatType
)
    opt_gap = 100abs(J - Jopt)/Jopt #Terminate by optimality gap

    return opt_gap < max_opt_gap
end

function sLME_ADMM_callback(
    z::Matrix{FloatType},
    w::Matrix{FloatType},
    α::Matrix{FloatType},
    v::Matrix{FloatType},
    β::Matrix{FloatType},
    iter::Int,
    J::FloatType
)
    opt_gap = 100abs(J - Jopt)/Jopt + 1e9norm(w .- v, Inf) #Terminate by optimality gap

    return opt_gap < max_opt_gap
end


function ADMM_callback_iter(
    z::Matrix{FloatType},
    w::Matrix{FloatType},
    α::Matrix{FloatType},
    iter::Int,
    J::FloatType,
    total_time::FloatType,
    cbs::callback_struct
)
    push!(cbs.rel_opt_gap, 100abs(J - Jopt)/Jopt)
    push!(cbs.n_iter, iter)

    return iter >= LADMM_n_iter
end

function sADMM_callback_iter(
    z::Matrix{FloatType},
    w::Matrix{FloatType},
    α::Matrix{FloatType},
    v::Matrix{FloatType},
    β::Matrix{FloatType},
    iter::Int,
    J::FloatType,
    cbs::callback_struct
)
    push!(cbs.rel_opt_gap, 100abs(J - Jopt)/Jopt)
    push!(cbs.n_iter, iter)
    return iter >= sLADMM_n_iter
end


function Ipopt_callback_iter(
    alg_mod::Cint,
    iter_count::Cint,
    obj_value::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls_trials::Cint,
    cbs::callback_struct
    )
    push!(cbs.rel_opt_gap, 100abs(Jopt - obj_value)/Jopt)
    push!(cbs.n_iter, iter_count)

    return (iter_count < Ipopt_n_iter)
end

# as entr_max, with Ipopt's and MadNLP's NLP scaling off
pick_solver(name, tol::FloatType = 1e-6, cbs::Union{Nothing, callback_struct} = nothing) =
    solver_model(name, tol; solvers = @__MODULE__, early_stop = Ipopt_callback_BM,
                 record = cbs === nothing ? nothing : (args...) -> Ipopt_callback_iter(args..., cbs),
                 scaling = false, madnlp_print_level = 5)
  