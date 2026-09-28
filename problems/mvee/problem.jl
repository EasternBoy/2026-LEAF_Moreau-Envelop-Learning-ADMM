using LinearAlgebra

const FloatType = Float64

function random_A_rotated(n, m)
    A = zeros(n,m)
    theta = randn() * pi
    c = cos(theta)
    s = sin(theta)
    
    Phi = Matrix{Float64}(I, n, n) # Start with Identity (Diagonal = 1s)
        Phi[1,1] =  c
        Phi[1,2] =  s
        Phi[2,1] = -s
        Phi[2,2] =  c
    count = 1
    while count <= m
        random_seed = 1
        rng = Random.MersenneTwister(1)
        S = randn(rng, Float64, n, n)
        S = S*Phi
        P = S'*S + 0.1*I(n)

        E, V = eigen(P)

        nE = clamp.(E, 0.3, 2.)

        nP = V'*diagm(nE)*V

        x = rand(Uniform(-1,1), n)

        if dot(x, nP, x) <= 1.
            @views A[:,count] = x
            count +=1
        end
    end 

    return A
end


mutable struct data_opt
    A::Matrix{FloatType}
    m::Int
    n::Int
    rho::FloatType
    cost_func::Function
end

function data_opt()
    n         = 3                       # dimension of X
    m         = 55                      # number of constraints
    A         = random_A_rotated(n, m)
    rho       = 1.
    cost_func = X -> logdet(X)
    return data_opt(A, m, n, rho, cost_func)
end

function pick_solver(name, tol = 1e-6)
    str = lowercase(name)
        if str == "mosek"
            model = Model(Mosek.Optimizer)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", tol)
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tol)
        # set_silent(solver)
        elseif str == "scs"
            model = Model(SCS.Optimizer)
            set_optimizer_attribute(model, "eps_abs", tol)
            set_optimizer_attribute(model, "eps_rel", tol)
        elseif str == "gurobi"
            model = Model(Gurobi.Optimizer)
        elseif str == "clarabel"
            model = Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "tol_feas", tol)
        else
            error("Unknown solver")
        end
    return model
end

