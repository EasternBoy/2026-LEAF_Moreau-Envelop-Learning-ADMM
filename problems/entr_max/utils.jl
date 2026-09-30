using LMEADMM   # src/LMEADMM.jl

# ICNN_MODEL, when defined before this file is included, picks another model (table.jl --model=...)
rho, mp = load_model(@isdefined(ICNN_MODEL) ? ICNN_MODEL : "models/entr_max/mEntropy-selfsupME-lw10-admm20-1144-rho=1-16.npz")
model   = ICNN(mp)
