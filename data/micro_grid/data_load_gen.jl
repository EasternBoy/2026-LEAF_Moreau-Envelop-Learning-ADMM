using CSV, DataFrames
using GaussianProcesses
using Optim
using Random, Statistics
using Pkg
using Plots

Pkg.activate(".")
Pkg.instantiate()

Random.seed!(42)

# ── 1. Load original 2-day building load data ──────────────────────────────────
# 192 samples at 15-min intervals, load ≤ 100 kW (San Diego building)
df_raw    = CSV.read("data/micro_grid/load_15min_max100kW_SanDiego_Building.csv", DataFrame)
load_orig = Float64.(df_raw[:, 2])

# ── 2. Extend to 7 days by tiling the daily pattern with small random noise ────
n_per_day    = 96      # 24 h × 4 samples/h
n_days       = 7
σ_day_noise  = 2.     # kW — small day-to-day variation

day_template = load_orig[1:n_per_day]   # day-1 profile used as the base template

load_7d = Float64[]
for _ in 1:n_days
    day = day_template .+ σ_day_noise .* randn(n_per_day)
    append!(load_7d, day)
end

n_total = length(load_7d)                                       # 672
t_h     = collect(range(0.0; step=0.25, length=n_total))       # hours: 0, 0.25, …, 167.75
df_7d = DataFrame(time_h=t_h, load_kw=load_7d)
CSV.write("data/micro_grid/load_7days.csv", df_7d)
println("Extended dataset ($(n_total) pts) saved → load_7days.csv")

# ── 3. Build and train Gaussian Process ────────────────────────────────────────
# Input x: (1 × n) matrix (GaussianProcesses.jl convention for 1-D inputs)
X = reshape(t_h, 1, n_total)
y = load_7d

# Kernel: Periodic with a fixed 24-hour period.
# Periodic(ll, lσ, lp)  — parameters on log-scale; lp = log(period [h])
period_h = 24.0
kern_periodic = fix(Periodic(log(1.0), log(1.0), log(period_h)), :lp)
kern = kern_periodic
# kern_trend    = SEIso(log(72.0), log(50.0))
# kern          = kern_periodic * kern_trend

mf = MeanConst(mean(y))
gp = GPE(X, y, mf, kern, log(σ_day_noise))

println("Initial log-likelihood : $(round(gp.mll; digits=3))")
optimize!(gp; method=LBFGS(), iterations=1000)
println("Optimised log-likelihood: $(round(gp.mll; digits=3))")
@assert isapprox(gp.kernel.kernel.p, period_h)
println("Fixed GP period         : $(gp.kernel.kernel.p) h")

# ── 4. Predict over the training horizon and save ──────────────────────────────
μ, σ² = predict_y(gp, X)
σ_pred = sqrt.(σ²)

df_gp = DataFrame(
    time_h   = t_h,
    load_kw  = load_7d,
    gp_mean  = μ,
    gp_std   = σ_pred,
    gp_lower = μ .- 2 .* σ_pred,
    gp_upper = μ .+ 2 .* σ_pred,
)
CSV.write("data/micro_grid/load_7days_gp.csv", df_gp)
println("GP predictions saved  → load_7days_gp.csv")

df_saved = CSV.read("data/micro_grid/load_7days_gp.csv", DataFrame)

plt = plot(
    df_saved.time_h,
    df_saved.gp_mean;
    ribbon=(df_saved.gp_mean .- df_saved.gp_lower, df_saved.gp_upper .- df_saved.gp_mean),
    fillalpha=0.2,
    color=:blue,
    linewidth=2,
    label="GP mean and 95% interval",
    xlabel="Time (h)",
    ylabel="Load (kW)",
    title="7-day building load",
    legend=:topright,
    grid=true,
    size=(1200, 500),
)
plot!(
    plt,
    df_saved.time_h,
    df_saved.load_kw;
    color=:black,
    linewidth=1,
    alpha=0.7,
    label="Saved load data",
)

plot_path = "data/micro_grid/load_7days_gp.png"
savefig(plt, plot_path)
println("7-day data plot saved → $(plot_path)")
