include("utils.jl")

if !(@isdefined(GUROBI_ENV))
    GUROBI_ENV = Gurobi.Env() 
end


J_opt::FloatType = 1.

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
    rel_opt_gap = 100abs((J_opt - obj_value/scale)/J_opt)
    stop = rel_opt_gap < max_opt_gap

    return !stop #False means running, True means stopping
end

@kwdef mutable struct callback_struct
    rel_opt_gap::Vector{FloatType} = FloatType[]
    n_iter::Vector{Int} = Int[]
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

function pick_solver(name, tol::FloatType = 1e-6, cbs::Union{Nothing, callback_struct} = nothing)
    #All solvers should share the same tolerance for stopping criteria
    str = lowercase(name)

    if str == "ipopt"      #for LP, QP, NLP
        model = Model(Ipopt.Optimizer)
        if tol >= 1e-3 # high tol implies low accuracy,
            set_optimizer_attribute(model, "tol",  1e-4) 
            MOI.set(model, Ipopt.CallbackFunction(), Ipopt_callback_BM) #set termination depends only on Ipopt_callback
        else
            set_optimizer_attribute(model, "tol",  tol) #High accuracy
            if cbs !== nothing
                cb_Ipopt = (args...) -> Ipopt_callback_iter(args..., cbs)
                MOI.set(model, Ipopt.CallbackFunction(), cb_Ipopt) #get data from Ipopt
            end
        end

    elseif str == "madnlp" #for NLP
        model = Model(()->MadNLP.Optimizer())
        MOI.set(model, MOI.Silent(), false)
        # set_optimizer_attribute(model, "print_level", 5) 

        set_optimizer_attribute(model, "blas_num_threads", nthreads())      
        set_optimizer_attribute(model, "tol", tol) 


    elseif str == "osqp"   #for LP, QP 
        model = Model(OSQP.Optimizer); 
        set_optimizer_attribute(model, "eps_abs", tol); 
        set_optimizer_attribute(model, "eps_rel", tol)

    elseif str == "gurobi" #for (MI)LP, (MI)QP, or (MI)NLP
        model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))

    elseif str == "clarabel" #for NLP
        model = Model(Clarabel.Optimizer)

    elseif str == "ecos" #for NLP
        model = Model(ECOS.Optimizer)

    else
        error("Specified solver is not supported")
    end

    set_silent(model)

    return model
end  