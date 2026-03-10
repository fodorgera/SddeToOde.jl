using SddeToOde
using DifferentialEquations
using Plots

# MATHIEU TEST

begin
    a1 = 0.2
    δ = 2.0
    b0 = -0.2
    ε = 1.0
    c0 = 0.15
    σ0 = 0.2
    ω = 1.0

    n = 2
    AM(t) = [0 1.0; -(δ + ε * cos(ω * t)) -a1]
    BM(t) = [0 0; b0 0]
    cM(t) = [0.0, c0]

    αM(t) = [0 0.0; -σ0*(δ+ε*cos(ω * t)) -σ0*a1]
    βM(t) = σ0 * BM(t)
    γM(t) = [0.0, σ0]

    τ = 2 * π
end;

# deterministic history φ(t) for t≤0
φ(t) = [0.0, 0.0];
T = 10;

mathieu_ode, mathieu_meta = SddeToOde.get_ode_from_sdde(AM, BM, cM, αM, βM, γM; τ=τ, T=T, φ=φ, m=10);

mathieu_sol = solve(mathieu_ode, Tsit5(), dt=τ/1000, adaptive=false);

mathieu_res = [SddeToOde.get_x_moments(mathieu_sol, mathieu_meta, t) for t in mathieu_sol.t];

mathieu_avgs = [r[1][1] for r in mathieu_res];
mathieu_vars = [r[2][1,1] for r in mathieu_res];

plot(mathieu_sol.t, [mathieu_avgs, mathieu_avgs .+ sqrt.(mathieu_vars)], label=["avg" "avg + std"])