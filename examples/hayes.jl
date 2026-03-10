using SddeToOde
using DifferentialEquations
using LinearAlgebra
using Plots

n = 1;
A = -6.0I(n);
B = 0.0*I(n);
c = [0.0];

α = 0.0I(n);
β = 2.0I(n);
γ = [1.0];

τ = 1.0;
T = 2.0;

φ(t) = [3.0];

hayes_ode, hayes_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=20);
# if you want DDEProblem outpu, use dde=true
hayes_dde, hayes_dde_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=20, dde=true);
# if you use DDEProblem output, use MethodOfSteps(Tsit5())
hayes_sol = solve(hayes_ode, Tsit5(), dt=τ/1000, adaptive=false);

hayes_res = [SddeToOde.get_x_moments(hayes_sol, hayes_meta, t) for t in hayes_sol.t];

avgs = [r[1][1] for r in hayes_res];
vars = [r[2][1,1] for r in hayes_res];

plot(hayes_sol.t, [avgs, avgs .+ sqrt.(vars)], label=["avg" "avg + std"])