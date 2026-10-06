using LMEADMM   # src/LMEADMM.jl

# ICNN_MODEL, when defined before this file is included, picks another model
rho, mp = load_model(@isdefined(ICNN_MODEL) ? ICNN_MODEL : "models/qp/qp-julia-moreau-rho=1.npz")
model   = ICNN(mp)

@assert rho ≈ qp_data["rho"][]
