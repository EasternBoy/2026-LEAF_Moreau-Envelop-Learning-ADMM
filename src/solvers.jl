# JuMP models for the reference solvers, shared by entr_max, power_grid and mvee
# (mpc configures its solvers differently, see problems/mpc/problem.jl).
# Each problem's setup.jl defines pick_solver(name, tol, cbs) on top of solver_model,
# passing its own Ipopt callbacks: the stopping rules differ per problem.
# The solver packages (Ipopt, Gurobi, ...) and GUROBI_ENV are taken from the module
# `solvers` that loaded them (setup.jl passes its own), so LMEADMM loads none of them.

"Optimality gaps and iteration counts recorded by the `*_callback_iter` callbacks."
@kwdef mutable struct callback_struct
    rel_opt_gap::Vector{FloatType} = FloatType[]
    n_iter::Vector{Int} = Int[]
end

"""
    solver_model(name, tol; solvers, early_stop, record, scaling, madnlp_print_level)

A silent JuMP model for solver `name`.  For Ipopt, `tol ≥ 1e-3` asks for a benchmark
solve that stops at a target: Ipopt's `tol` is 1e-4 and the callback `early_stop`
decides when to stop; a smaller `tol` is used as is, with the callback `record`
(if any) installed to record the iterates.  `scaling = false` switches Ipopt's and
MadNLP's NLP scaling off.  The solver package is looked up in the module `solvers`.
"""
function solver_model(name, tol; solvers::Module = Main, early_stop = nothing, record = nothing,
                      scaling::Bool = true, madnlp_print_level = nothing)
    str = lowercase(name)
    pkg(s) = getfield(solvers, s)

    if str == "ipopt"      #for LP, QP, NLP
        Ipopt = pkg(:Ipopt)
        model = Model(Ipopt.Optimizer)
        scaling || set_optimizer_attribute(model, "nlp_scaling_method", "none")
        if tol >= 1e-3 # high tol implies low accuracy,
            set_optimizer_attribute(model, "tol",  1e-4)
            MOI.set(model, Ipopt.CallbackFunction(), early_stop) #set termination depends only on Ipopt_callback
        else
            set_optimizer_attribute(model, "tol",  tol) #High accuracy
            record === nothing || MOI.set(model, Ipopt.CallbackFunction(), record) #get data from Ipopt
        end

    elseif str == "madnlp" #for NLP
        MadNLP = pkg(:MadNLP)
        model = Model(()->MadNLP.Optimizer())
        MOI.set(model, MOI.Silent(), false)
        madnlp_print_level === nothing || set_optimizer_attribute(model, "print_level", madnlp_print_level)
        set_optimizer_attribute(model, "blas_num_threads", nthreads())
        set_optimizer_attribute(model, "tol", tol)
        scaling || set_optimizer_attribute(model, "nlp_scaling", false)

    elseif str == "osqp"   #for LP, QP
        model = Model(pkg(:OSQP).Optimizer);
        set_optimizer_attribute(model, "eps_abs", tol);
        set_optimizer_attribute(model, "eps_rel", tol)

    elseif str == "gurobi" #for (MI)LP, (MI)QP, or (MI)NLP
        Gurobi, env = pkg(:Gurobi), pkg(:GUROBI_ENV)
        model = Model(() -> Gurobi.Optimizer(env))

    elseif str == "clarabel" #for NLP
        model = Model(pkg(:Clarabel).Optimizer)

    elseif str == "ecos" #for NLP
        model = Model(pkg(:ECOS).Optimizer)

    elseif str == "mosek"
        model = Model(pkg(:Mosek).Optimizer)
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", tol)
        set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tol)

    else
        error("Specified solver is not supported")
    end

    set_silent(model)

    return model
end
