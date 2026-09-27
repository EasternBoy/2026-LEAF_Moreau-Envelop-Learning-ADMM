include(joinpath(@__DIR__, "..", "..", "src", "icnn.jl"))    # ICNN, load_model, gradient_struct, mini_batch

rho, mp = load_model("models/entr_max/mEntropy-rho=1-16.json")
model   = ICNN(mp)
