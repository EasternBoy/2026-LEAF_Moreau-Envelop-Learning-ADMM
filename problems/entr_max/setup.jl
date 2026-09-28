include("utils.jl")
using LMEADMM   # src/LMEADMM.jl

if !(@isdefined(GUROBI_ENV))
    GUROBI_ENV = Gurobi.Env() 
end


J_opt::FloatType = 1.
ipopt_feas_tol::FloatType = 1e-4   # early-stop feasibility tolerance on w (see Ipopt_callback_BM)

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
   scale,
)
    rel_opt_gap = 100abs((J_opt - obj_value/scale)/J_opt)
    # also require primal feasibility: inf_pr is the constraint violation in x = scale*w,
    # so inf_pr < ipopt_feas_tol*scale means a violation below ipopt_feas_tol in w
    stop = rel_opt_gap < max_opt_gap && inf_pr < ipopt_feas_tol*scale

    return !stop #False means running, True means stopping
end



function ADMM_callback(
    z::VecOrMat{FloatType},
    w::VecOrMat{FloatType},
    α::VecOrMat{FloatType},
    iter::Int,
    J::FloatType,
    sol_time_upto_iter::FloatType
)
    opt_gap = 100abs(J - J_opt)/abs(J_opt) #Terminate by optimality gap

    return opt_gap < max_opt_gap
end

function sLME_ADMM_callback(
    z::VecOrMat{FloatType},
    w::VecOrMat{FloatType},
    α::VecOrMat{FloatType},
    v::VecOrMat{FloatType},
    β::VecOrMat{FloatType},
    iter::Int,
    J::FloatType
)
    opt_gap = 100abs(J - J_opt)/abs(J_opt) #+ 1e9norm(w .- v, Inf) #Terminate by optimality gap

    return opt_gap < max_opt_gap
end


function ADMM_callback_iter(
    z::VecOrMat{FloatType},
    w::VecOrMat{FloatType},
    α::VecOrMat{FloatType},
    iter::Int,
    J::FloatType,
    total_time::FloatType,
    cbs::callback_struct
)
    push!(cbs.rel_opt_gap, 100abs((J - J_opt)/J_opt))
    push!(cbs.n_iter, iter)

    return iter >= 100
end

function sADMM_callback_iter(
    z::VecOrMat{FloatType},
    w::VecOrMat{FloatType},
    α::VecOrMat{FloatType},
    v::VecOrMat{FloatType},
    β::VecOrMat{FloatType},
    iter::Int,
    J::FloatType,
    cbs::callback_struct
)
    push!(cbs.rel_opt_gap, 100abs((J - J_opt)/J_opt))
    push!(cbs.n_iter, iter)
    return iter >= 1000
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
    push!(cbs.rel_opt_gap, 100abs(J_opt - obj_value)/J_opt)
    push!(cbs.n_iter, iter_count)

    return (iter_count < Ipopt_n_iter)
end

# Ipopt stops on Ipopt_callback_BM for benchmark solves (tol ≥ 1e-3), records with Ipopt_callback_iter otherwise
# (it needs the problem's variable scale, see var_scale)
pick_solver(name, tol::FloatType = 1e-6, cbs::Union{Nothing, callback_struct} = nothing; scale = nothing) =
    solver_model(name, tol; solvers = @__MODULE__, early_stop = scale === nothing ? nothing : (args...) -> Ipopt_callback_BM(args..., scale),
                 record = cbs === nothing ? nothing : (args...) -> Ipopt_callback_iter(args..., cbs))
  