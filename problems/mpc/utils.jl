using LMEADMM   # src/LMEADMM.jl

rho, mp = load_model("models/mpc/test.json")
model   = ICNN(mp)
