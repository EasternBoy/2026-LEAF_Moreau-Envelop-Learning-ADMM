include("utils.jl")


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


@kwdef mutable struct callback_struct
    rel_opt_gap::Vector{FloatType} = FloatType[]
    n_iter::Vector{Int} = Int[]
end


function sLME_ADMM_callback( J::FloatType)
    opt_gap = 100abs(J - Jopt)/Jopt 

    return opt_gap < max_opt_gap
end


function pick_solver(name, tol::FloatType = 1e-6, cbs::Union{Nothing, callback_struct} = nothing)
    #All solvers should share the same tolerance for stopping criteria
    str = lowercase(name)

    if str == "ipopt"      #for LP, QP, NLP
        model = Model(Ipopt.Optimizer)
        if tol >= 1e-3 # high tol implies low accuracy,
            set_optimizer_attribute(model, "nlp_scaling_method", "none") #set none scaling method
            set_optimizer_attribute(model, "tol",  1e-4) 
            MOI.set(model, Ipopt.CallbackFunction(), Ipopt_callback_BM) #set termination depends only on Ipopt_callback
        else
            set_optimizer_attribute(model, "nlp_scaling_method", "none") #set none scaling method
            set_optimizer_attribute(model, "tol",  tol) #High accuracy
            if cbs !== nothing
                cb_Ipopt = (args...) -> Ipopt_callback_iter(args..., cbs)
                MOI.set(model, Ipopt.CallbackFunction(), cb_Ipopt) #set termination depends only on Ipopt_callback
            end
        end

    elseif str == "madnlp" #for NLP
        model = Model(()->MadNLP.Optimizer())
        MOI.set(model, MOI.Silent(), false)
        set_optimizer_attribute(model, "print_level", 5) 

        set_optimizer_attribute(model, "blas_num_threads", nthreads())      
        set_optimizer_attribute(model, "tol", tol) 
        set_optimizer_attribute(model, "nlp_scaling", false)


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
    elseif str == "mosek"
            model = Model(Mosek.Optimizer)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", tol)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tol)

    else
        error("Specified solver is not supported")
    end

    set_silent(model)

    return model
end  