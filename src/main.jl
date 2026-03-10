# Helpers to index blocks
@inline function blockrange(n::Int, j::Int)
    # j = 0..m
    (j*n + 1):((j+1)*n)
end

"""
Build the time-independent delay-line drift (shift chain) for the upwind transport approximation.

z = [x0; x1; ...; xm], each xj ∈ R^n, N = n*(m+1)
This contains only the xj' = (1/h)(x_{j-1} - xj) part for j = 1..m.
"""
function build_delay_chain_drift(n::Int, τ; m::Int)
    @assert m ≥ 1

    h = τ/m
    N = n*(m+1)

    Ae = zeros(Float64, N, N)
    ce = zeros(Float64, N)

    I_n = Matrix{Float64}(I, n, n)
    @inbounds for j in 1:m
        rj = blockrange(n, j)
        if 1 < j < m-1
            rjm1 = blockrange(n, j-1)
            rjp1 = blockrange(n, j+1)
            rjm2 = blockrange(n, j-2)
            rjp2 = blockrange(n, j+2)
            Ae[rj, rjm2] .+= (-1/(12h)) .* I_n
            Ae[rj, rjp2] .+= (1/(12h)) .* I_n
            Ae[rj, rjm1] .+= (8/(12h)) .* I_n
            Ae[rj, rjp1] .+= (-8/(12h)) .* I_n
        elseif j < m
            rjm1 = blockrange(n, j-1)
            rjp1 = blockrange(n, j+1)
            Ae[rj, rjm1] .+= (1/(2h)) .* I_n
            Ae[rj, rjp1] .+= (-1/(2h)) .* I_n
        else
            rjm1 = blockrange(n, j-1)
            Ae[rj, rj]   .+= (-1/h) .* I_n
            Ae[rj, rjm1] .+= (1/h)  .* I_n
        end
    end

    return Ae, ce, h
end

"""
Build full extended drift Ae and ce for constant A,B,c by adding the head block
to the time-independent delay-chain drift.
"""
function build_extended_drift(A, B, c, τ; m::Int)
    n = size(A, 1)
    @assert size(A,2)==n
    @assert size(B,1)==n && size(B,2)==n
    @assert length(c)==n

    Ae_delay, ce, h = build_delay_chain_drift(n, τ; m=m)
    N = n*(m+1)

    Ae = similar(Ae_delay)
    Ae .= Ae_delay

    r0 = blockrange(n, 0)
    rm = blockrange(n, m)
    Ae[r0, r0] .= A
    Ae[r0, rm] .+= B
    ce[r0] .= c

    return Ae, ce, h
end

"""
Compute E[g g'] where g = α x0 + β xm + γ
using μ and covariance P of the extended state.
"""
function EgEgT_from_muP(α, β, γ, μ0, μm, P00, Pmm, P0m)
    # raw second moments
    Ex0x0 = P00 .+ μ0*μ0'
    Exmxm = Pmm .+ μm*μm'
    Ex0xm = P0m .+ μ0*μm'
    Exmx0 = Ex0xm'

    G = α*Ex0x0*α' + α*Ex0xm*β' + β*Exmx0*α' + β*Exmxm*β'
    G .+= α*μ0*γ' + γ*μ0'*α'
    G .+= β*μm*γ' + γ*μm'*β'
    G .+= γ*γ'
    return G
end

function EgEgT_from_muP!(G, α, β, γ, μ0, μm, P00, Pmm, P0m)
    G .= EgEgT_from_muP(α, β, γ, μ0, μm, P00, Pmm, P0m)
    return nothing
end

"""
Solve mean/covariance ODE for the extended-state approximation.

Coefficients A, B, c, α, β, γ can be constant arrays or functions of time t.

Returns:
  sol: ODESolution over state y = [vec(μ); vec(P)]
Extraction:
  N = n*(m+1)
  μ(t) = reshape(sol(t)[1:N], N)
  P(t) = reshape(sol(t)[N+1:end], N, N)

Mean/variance of original x(t):
  μx(t) = μ0 block (j=0)
  Vx(t) = P00 block (j=0,j=0)
"""
function get_ode_from_sdde(A, B, c, α, β, γ; τ, T, φ, m::Int=200,
                                tspan=(0.0,T),dde=false, kwargs...)
    # allow both constant and time-dependent coefficients
    getA(t) = A isa Function ? A(t) : A
    getB(t) = B isa Function ? B(t) : B
    getc(t) = c isa Function ? c(t) : c
    getα(t) = α isa Function ? α(t) : α
    getβ(t) = β isa Function ? β(t) : β
    getγ(t) = γ isa Function ? γ(t) : γ

    A0 = getA(0.0)
    n = size(A0, 1)

    Ae_delay, ce_delay, h = build_delay_chain_drift(n, τ; m=m)
    N = n*(m+1)

    # Initial mean μ(0): fill delay line with history φ on [-τ,0]
    # xj(0) ≈ φ(-j*h)
    μ0 = zeros(eltype(getc(0.0)), N)
    for j in 0:m
        μ0[blockrange(n,j)] .= φ(-j*h)
    end

    # Initial covariance P(0): typically zero if deterministic history
    P0 = zeros(eltype(getc(0.0)), N, N)

    y0 = vcat(vec(μ0), vec(P0))

    # Preallocate workspaces for the ODE RHS to avoid per-step allocations
    r0 = blockrange(n, 0)
    rm = blockrange(n, m)
    Ae = similar(Ae_delay)
    AeT = similar(Ae_delay)
    ce = similar(ce_delay)
    tmpP1 = similar(P0)
    tmpP2 = similar(P0)
    Q = similar(P0)
    EgEg = zeros(eltype(P0), n, n)
    zeroP = zero(eltype(P0))

    # function f!(dy, y, hfun, p, t)
    function f!(dy, y, p, t)
        @views μ = y[1:N]
        @views P = reshape(y[N+1:end], N, N)

        @views dμ = dy[1:N]
        @views dP = reshape(dy[N+1:end], N, N)

        @inbounds begin
            # rebuild time-dependent Ae(t), ce(t) from delay chain + current A(t),B(t),c(t)
            Ae .= Ae_delay
            ce .= ce_delay

            At = getA(t)
            Bt = getB(t)
            ct = getc(t)

            @views Ae[r0, r0] .= At
            @views Ae[r0, rm] .+= Bt
            @views ce[r0] .= ct

            # dμ = Ae(t) * μ + ce(t)
            mul!(dμ, Ae, μ)
            dμ .+= ce

            # pull blocks for diffusion term (only affects x0 block)
            @views μ_0 = μ[r0]
            @views μ_m = μ[rm]

            @views P00 = P[r0, r0]
            @views Pmm = P[rm, rm]
            @views P0m = P[r0, rm]

            # compute EgEg in-place into workspace with time-dependent α,β,γ
            EgEgT_from_muP!(EgEg, getα(t), getβ(t), getγ(t), μ_0, μ_m, P00, Pmm, P0m)

            # Q only has nonzero entries in the (0,0) block; reuse workspace
            fill!(Q, zeroP)
            @views Q[r0, r0] .= EgEg

            # dP = Ae(t)*P + P*Ae(t)' + Q, computed with workspaces
            AeT .= transpose(Ae)
            mul!(tmpP1, Ae, P)
            mul!(tmpP2, P, AeT)
            dP .= tmpP1
            dP .+= tmpP2
            dP .+= Q
        end

        return nothing
    end

    f_dde!(dy, y, h, p, t) = f!(dy, y, p, t)

    hist(p, s) = y0;

    prob = dde ? DDEProblem(f_dde!, y0, hist, tspan) : ODEProblem(f!, y0, tspan)

    meta = (Ae=Ae, ce=ce, h=h, N=N, n=n, m=m);

    return prob, meta
end

"""
Helper to extract μx(t), Vx(t) from solution.
"""
function get_x_moments(sol, meta, t)
    N,n,m = meta.N, meta.n, meta.m
    y = sol(t)
    μ = reshape(view(y, 1:N), N)
    P = reshape(view(y, N+1:length(y)), N, N)

    r0 = blockrange(n,0)
    μx = μ[r0]
    Vx = P[r0, r0]
    return μx, Vx
end