using LMEADMM   # src/LMEADMM.jl

rho, mp = load_model("models/entr_max/mEntropy-rho=1-16.json")
model   = ICNN(mp)
