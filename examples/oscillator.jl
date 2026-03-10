using SddeToOde
using DifferentialEquations
using Plots

# OSCILLATOR TEST

n = 2;
p = 7; d = 3.5;
σ0 = 0.1;
A = [0.0 1.0; 5.0 0.0];
B = [0.0 0.0; -p -d];
β = σ0*B;
γ = [0.0, σ0];
τ = 0.3;
c = [0.0, 0.0];
α = [0.0 0.0; 0.0 0.0];

φ(t) = [0.1, 0.0];

T = 3.0;

osc_ode, osc_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=20);
# if you want DDEProblem outpu, use dde=true
osc_dde, osc_dde_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=20, dde=true);

# if you use DDEProblem output, use MethodOfSteps(Tsit5())
osc_sol = solve(osc_ode, Tsit5(), dt=τ/1000, adaptive=false);

osc_res = [SddeToOde.get_x_moments(osc_sol, osc_meta, t) for t in osc_sol.t];

osc_avgs = [r[1][1] for r in osc_res];
osc_vars = [r[2][1,1] for r in osc_res];

plot(osc_sol.t, [osc_avgs, osc_avgs .+ sqrt.(osc_vars)], label=["avg" "avg + std"])