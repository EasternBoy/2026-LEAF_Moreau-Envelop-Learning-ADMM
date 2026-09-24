using Test
# Pass a small exported instance directory as ARGS[1]. This also smoke-tests the driver.
include(joinpath(@__DIR__, "..", "julia", "baselines_power.jl"))

@testset "Power LME reset and feasibility" begin
    init = FloatType(x0s[1]); load = Vector{FloatType}(loads[1,:]); gen = Vector{FloatType}(gens[1,:])
    first, _ = admm(init, load, gen; max_iter=3)
    first = copy(first)
    admm(FloatType(x0s[end]), Vector{FloatType}(loads[end,:]), Vector{FloatType}(gens[end,:]); max_iter=2)
    repeated, _ = admm(init, load, gen; max_iter=3)
    @test first == repeated
    # A successful objective callback must not bypass an infeasible iterate.
    seen = Ref(0)
    cb = (args...) -> (seen[] = args[6]; true)
    short, _ = admm(init, load, gen, cb; max_iter=3)
    @test seen[] == 3 || eco_solution_feasible(mpc_data, short, init, load, gen, 1e-4)
    invalid = copy(short); invalid[3,1] = -1e-8
    @test !eco_solution_feasible(mpc_data, invalid, init, load, gen, 1e-4)
    @test !power_metrics(mpc_data, invalid, init, load, gen).valid
    @test isnan(cone_metrics(zeros(1,2), ones(1), [-1e-5,1+1e-5]).obj)
end
