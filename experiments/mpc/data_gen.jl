using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Printf
using LinearAlgebra
using NPZ, Clarabel, JuMP

include("../../problems/mpc/problem.jl")
include("../../problems/mpc/jump_solver.jl")
include("../../problems/mpc/admm.jl")

# ── Simulation parameters ──
n_sim        = 10     # closed-loop MPC steps per trajectory
n_traj_train = 20      # training trajectories
n_traj_test  = 2       # test trajectories

para_opt    = MPCData()
solver_name = "Clarabel"

const scale = 1.0


function step_mpc(para::MPCData, u1::Vector{FloatType})
    x_next = para.A * para.x0 + para.B * u1
    return MPCData(para.A, para.B, para.Q, para.Qt, para.R,
                   para.rho, para.x_min, para.x_max,
                   para.u_min, para.u_max,
                   para.nx, para.nu, para.T, x_next, para.cost_func)
end


function process_and_save(raw::Dict, fname::String)
    X = reduce(hcat, raw["input"])   # (m, N)
    G = reduce(hcat, raw["grad"])    # (m, N)
    E = raw["env"]                   # (N,)

    keep = [norm(G[:, j]) > 1e-6 for j in axes(G, 2)]
    X, G, E = X[:, keep], G[:, keep], E[keep]
    @printf("  kept %d / %d points after zero-grad filter\n", sum(keep), length(keep))

    data = Dict(
        "input" => X ./ scale,             
        "grad"  => G .* scale,             
        "rho"   => para_opt.rho,
        "enve"  => E,
        "scale" => scale,
    )
    npzwrite(fname, data)
end

# ── Build solvers once ──
admm_sol = ADMM_mpc(para_opt, solver_name; tol=1e-3)

# ── Training dataset: closed-loop simulation ──
data_train = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])

for traj in 1:n_traj_train
    current = new_instance(para_opt)
    println("=== Training trajectory $traj / $n_traj_train ===")

    for t in 1:n_sim
        _, _, J_opt = JuMP_solver(solver_name, current, 1e-6)
        u1, J_ADMM  = admm_sol(data_train, current, verbose=true)

        @printf("  step %3d/%d  J_opt = %8.4f  J_ADMM = %8.4f  ΔJ/J = %5.3f%%\n",
                t, n_sim, J_opt, J_ADMM, 100abs(J_opt - J_ADMM) / abs(J_opt))

        next = step_mpc(current, u1)
        if next === nothing
            println("  [stop] x left feasible region at step $t — ending trajectory early")
            break
        end
        current = next
    end
end

@printf("\nCollected %d raw training data points\n", length(data_train["env"]))
process_and_save(data_train,
    joinpath("data", "mpc", "training", string("mpc-train-rho=", para_opt.rho, ".npz")))


# ── Test dataset: closed-loop simulation ──
data_test = Dict("input" => Vector{FloatType}[], "env" => FloatType[], "grad" => Vector{FloatType}[])

for traj in 1:n_traj_test
    current = new_instance(para_opt)
    println("=== Test trajectory $traj / $n_traj_test ===")

    for t in 1:n_sim
        _, _, J_opt = JuMP_solver(solver_name, current, 1e-6)
        u1, J_ADMM  = admm_sol(data_test, current, verbose=true)

        @printf("  step %3d/%d  J_opt = %8.4f  J_ADMM = %8.4f  ΔJ/J = %5.3f%%\n",
                t, n_sim, J_opt, J_ADMM, 100abs(J_opt - J_ADMM) / abs(J_opt))

        next = step_mpc(current, u1)
        if next === nothing
            println("  [stop] x left feasible region at step $t — ending trajectory early")
            break
        end
        current = next
    end
end

@printf("\nCollected %d raw test data points\n", length(data_test["env"]))
process_and_save(data_test,
    joinpath("data", "mpc", "training", string("mpc-test-rho=", para_opt.rho, ".npz")))
