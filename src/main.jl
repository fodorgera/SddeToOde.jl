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
The delay τ can be a constant or a function of time t.

When τ is time-dependent, `τ_max` must be supplied to fix the delay-line grid.
The delay chain is discretized on [0, τ_max] with m nodes (spacing h = τ_max/m),
and x(t - τ(t)) is obtained by linear interpolation between the two nearest grid
nodes.  The interpolated covariance blocks are:

    μ_τ  = (1-s) μ_j + s μ_{j+1}
    P_ττ = (1-s)² P_{jj} + (1-s)s (P_{j,j+1} + P_{j+1,j}) + s² P_{j+1,j+1}
    P_0τ = (1-s) P_{0j} + s P_{0,j+1}

where j = floor(τ(t)/h), s = (τ(t) - j*h)/h.

Returns:
  prob: ODEProblem (or DDEProblem if dde=true) over state y = [vec(μ); vec(P)]
  meta: NamedTuple with (Ae, ce, h, N, n, m, τ_max)

Mean/variance of original x(t):
  μx(t) = μ0 block (j=0)
  Vx(t) = P00 block (j=0,j=0)
"""
function get_ode_from_sdde(A, B, c, α, β, γ; τ, T, φ, m::Int=10,
                                tspan=(0.0,T), dde=false, τ_max=nothing, kwargs...)
    getA(t,x,xτ) = parseFunction(A,t,x,xτ)
    getB(t,x,xτ) = parseFunction(B,t,x,xτ)
    getc(t,x,xτ) = parseFunction(c,t,x,xτ)
    getα(t,x,xτ) = parseFunction(α,t,x,xτ)
    getβ(t,x,xτ) = parseFunction(β,t,x,xτ)
    getγ(t,x,xτ) = parseFunction(γ,t,x,xτ)

    τ_is_func = τ isa Function
    getτ(t) = τ_is_func ? τ(t) : τ

    # τ_grid sets the delay-line extent; must be constant for a fixed state size
    τ_grid = if τ_max !== nothing
        Float64(τ_max)
    elseif τ_is_func
        error("τ_max must be provided when τ is time-dependent")
    else
        Float64(τ)
    end

    use_interp = τ_is_func || τ_max !== nothing

    A0 = getA(0.0, φ(0.0), φ(0.0))
    n = size(A0, 1)

    Ae_delay, ce_delay, h = build_delay_chain_drift(n, τ_grid; m=m)
    N = n*(m+1)

    # xj(0) ≈ φ(-j*h)
    μ0 = zeros(eltype(getc(0.0, φ(0.0), φ(0.0))), N)
    for j in 0:m
        μ0[blockrange(n,j)] .= φ(-j*h)
    end

    P0 = zeros(eltype(getc(0.0, φ(0.0), φ(0.0))), N, N)

    y0 = vcat(vec(μ0), vec(P0))

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

    # Workspaces for interpolated delayed-state moments
    μ_del  = zeros(eltype(P0), n)
    P_del  = zeros(eltype(P0), n, n)
    P_0del = zeros(eltype(P0), n, n)

    function f!(dy, y, p, t)
        @views μ = y[1:N]
        @views P = reshape(y[N+1:end], N, N)

        @views dμ = dy[1:N]
        @views dP = reshape(dy[N+1:end], N, N)

        @inbounds begin
            Ae .= Ae_delay
            ce .= ce_delay
            # todo have to select the correct block of y for the function calls
            
            
            # states for y and yτ
            y0 = μ[r0]
            yτ = μ[rm]
            At = getA(t, y0, yτ)
            Bt = getB(t, y0, yτ)
            ct = getc(t, y0, yτ)

            @views Ae[r0, r0] .= At
            @views ce[r0] .= ct

            if use_interp
                τt = clamp(getτ(t), 0.0, τ_grid)
                jd = min(floor(Int, τt / h), m - 1)
                s  = (τt - jd * h) / h
                s1 = 1.0 - s

                rj  = blockrange(n, jd)
                rjp = blockrange(n, jd + 1)

                @views Ae[r0, rj]  .+= s1 .* Bt
                @views Ae[r0, rjp] .+= s  .* Bt

                @views μ_del .= s1 .* μ[rj] .+ s .* μ[rjp]

                @views P_del .= s1^2 .* P[rj, rj] .+
                                s1*s .* (P[rj, rjp] .+ P[rjp, rj]) .+
                                s^2  .* P[rjp, rjp]

                @views P_0del .= s1 .* P[r0, rj] .+ s .* P[r0, rjp]

                EgEgT_from_muP!(EgEg, getα(t, y0, yτ), getβ(t, y0, yτ), getγ(t, y0, yτ),
                                view(μ, r0), μ_del,
                                view(P, r0, r0), P_del, P_0del)
            else
                @views Ae[r0, rm] .+= Bt

                EgEgT_from_muP!(EgEg, getα(t, y0, yτ), getβ(t, y0, yτ), getγ(t, y0, yτ),
                                view(μ, r0), view(μ, rm),
                                view(P, r0, r0), view(P, rm, rm), view(P, r0, rm))
            end

            mul!(dμ, Ae, μ)
            dμ .+= ce

            fill!(Q, zeroP)
            @views Q[r0, r0] .= EgEg

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

    meta = (Ae=Ae, ce=ce, h=h, N=N, n=n, m=m, τ_max=τ_grid)

    return prob, meta
end

function parseFunction(f,t,args...)
    if f isa Function
        try
            return f(t,args...)
        catch
            return f(t)
        end
    else
        return f
    end
end


function get_ode_from_nsdde(f,fx,fy,g,gx,gy; kwargs...)
    A(t,x,xτ) = fx(t,x,xτ)
    B(t,x,xτ) = fy(t,x,xτ)
    c(t,x,xτ) = f(t,x,xτ) .- A(t,x,xτ)*x .- B(t,x,xτ)*xτ
    α(t,x,xτ) = gx(t,x,xτ)
    β(t,x,xτ) = gy(t,x,xτ)
    γ(t,x,xτ) = g(t,x,xτ) .- α(t,x,xτ)*x .- β(t,x,xτ)*xτ
    return get_ode_from_sdde(A, B, c, α, β, γ; kwargs...)
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