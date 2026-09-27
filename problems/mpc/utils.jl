include(joinpath(@__DIR__, "..", "..", "src", "icnn.jl"))    # ICNN, load_model, gradient_struct, mini_batch

rho, mp = load_model("models/mpc/test.json")
model   = ICNN(mp)
