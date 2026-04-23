# Helpers to index blocks
@inline function blockrange(n::Int, j::Int)
    # j = 0..m
    (j*n + 1):((j+1)*n)
end

"""
Build the time-independent delay-line drift (shift chain) for the upwind transport approximation.

z = [x0; x1; ...; xm], each xj ∈ R^n, N = n*(m+1)
This contains only the xj' = (1/h)(x_{j-1} - xj) part for j = 1..m.

Returned as a `SparseMatrixCSC` since the pattern consists only of a few block-diagonal bands.
"""
function build_delay_chain_drift(n::Int, τ; m::Int)
    @assert m ≥ 1

    h = τ/m
    N = n*(m+1)

    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]

    @inbounds for j in 1:m
        rj = blockrange(n, j)
        if 1 < j < m-1
            rjm1 = blockrange(n, j-1)
            rjp1 = blockrange(n, j+1)
            rjm2 = blockrange(n, j-2)
            rjp2 = blockrange(n, j+2)
            for k in 1:n
                push!(I_idx, rj[k]); push!(J_idx, rjm2[k]); push!(V, -1/(12h))
                push!(I_idx, rj[k]); push!(J_idx, rjp2[k]); push!(V,  1/(12h))
                push!(I_idx, rj[k]); push!(J_idx, rjm1[k]); push!(V,  8/(12h))
                push!(I_idx, rj[k]); push!(J_idx, rjp1[k]); push!(V, -8/(12h))
            end
        elseif j < m
            rjm1 = blockrange(n, j-1)
            rjp1 = blockrange(n, j+1)
            for k in 1:n
                push!(I_idx, rj[k]); push!(J_idx, rjm1[k]); push!(V,  1/(2h))
                push!(I_idx, rj[k]); push!(J_idx, rjp1[k]); push!(V, -1/(2h))
            end
        else
            rjm1 = blockrange(n, j-1)
            for k in 1:n
                push!(I_idx, rj[k]); push!(J_idx, rj[k]);   push!(V, -1/h)
                push!(I_idx, rj[k]); push!(J_idx, rjm1[k]); push!(V,  1/h)
            end
        end
    end

    Ae = sparse(I_idx, J_idx, V, N, N)
    ce = zeros(Float64, N)

    return Ae, ce, h
end

"""
Build full extended drift Ae and ce for constant A,B,c by adding the head block
to the time-independent delay-chain drift. Returned Ae is sparse.
"""
function build_extended_drift(A, B, c, τ; m::Int)
    n = size(A, 1)
    @assert size(A,2)==n
    @assert size(B,1)==n && size(B,2)==n
    @assert length(c)==n

    Ae_delay, ce, h = build_delay_chain_drift(n, τ; m=m)
    N = n*(m+1)

    # Add head block contributions to a (sparse) copy of Ae_delay.
    Ae = copy(Ae_delay)

    r0 = blockrange(n, 0)
    rm = blockrange(n, m)
    # A on the (r0, r0) block (replace, since delay chain has no entries there)
    for i in 1:n, k in 1:n
        Ae[r0[i], r0[k]] = A[i, k]
    end
    # +B on the (r0, rm) block
    for i in 1:n, k in 1:n
        Ae[r0[i], rm[k]] += B[i, k]
    end
    ce[r0] .= c

    dropzeros!(Ae)
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

Internally the drift is kept as a sparse `Ae_delay` (only the delay-chain bands)
plus a small dense head correction on the first `n` rows / columns.  The dense
`N×N` drift is never assembled, and the expensive products `Ae*P`, `P*Ae'` are
computed as `Ae_delay*P + head` (sparse-times-dense), saving a lot of memory
traffic when `m` is large.

Returns:
  prob: ODEProblem (or DDEProblem if dde=true) over state y = [vec(μ); vec(P)]
  meta: NamedTuple with (Ae_delay, ce, h, N, n, m, τ_max)

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

    # Precompute the transpose once; kept as a sparse (adjoint) handle.
    Ae_delayT = sparse(transpose(Ae_delay))

    # xj(0) ≈ φ(-j*h)
    T0 = eltype(getc(0.0, φ(0.0), φ(0.0)))
    μ0 = zeros(T0, N)
    for j in 0:m
        μ0[blockrange(n,j)] .= φ(-j*h)
    end

    P0 = zeros(T0, N, N)

    y0 = vcat(vec(μ0), vec(P0))

    r0 = blockrange(n, 0)
    rm = blockrange(n, m)

    # Workspaces
    EgEg    = zeros(T0, n, n)
    μ_del   = zeros(T0, n)
    P_del   = zeros(T0, n, n)
    P_0del  = zeros(T0, n, n)
    # n×N and N×n buffers for the dense head row / column of the drift
    head_row = zeros(T0, n, N)  # (At*P[r0,:] + Bt*P_row_del)
    head_col = zeros(T0, N, n)  # (P[:,r0]*At' + P_col_del*Bt')
    P_row_del = zeros(T0, n, N) # s1*P[rj,:] + s*P[rjp,:]
    P_col_del = zeros(T0, N, n) # s1*P[:,rj] + s*P[:,rjp]

    function f!(dy, y, p, t)
        @views μ = y[1:N]
        @views P = reshape(y[N+1:end], N, N)

        @views dμ = dy[1:N]
        @views dP = reshape(dy[N+1:end], N, N)

        @inbounds begin
            y0v = μ[r0]
            yτv = μ[rm]
            At = getA(t, y0v, yτv)
            Bt = getB(t, y0v, yτv)
            ct = getc(t, y0v, yτv)

            # Resolve the two delay-line grid nodes (rj, rjp) and interp weights.
            if use_interp
                τt = clamp(getτ(t), 0.0, τ_grid)
                jd = min(floor(Int, τt / h), m - 1)
                s  = (τt - jd * h) / h
            else
                # Pure "Bt at rm" case == (jd=m-1, s=1).
                jd = m - 1
                s  = 1.0
            end
            s1 = 1.0 - s

            rj  = blockrange(n, jd)
            rjp = blockrange(n, jd + 1)

            # Interpolated delayed-state moments.
            @views μ_del .= s1 .* μ[rj] .+ s .* μ[rjp]
            @views P_del .= s1^2 .* P[rj, rj] .+
                            s1*s .* (P[rj, rjp] .+ P[rjp, rj]) .+
                            s^2  .* P[rjp, rjp]
            @views P_0del .= s1 .* P[r0, rj] .+ s .* P[r0, rjp]

            EgEgT_from_muP!(EgEg, getα(t, y0v, yτv), getβ(t, y0v, yτv), getγ(t, y0v, yτv),
                            view(μ, r0), μ_del,
                            view(P, r0, r0), P_del, P_0del)

            # --- dμ = Ae * μ + ce  ------------------------------------------
            # Ae = Ae_delay + H, with H supported only on rows r0.
            # So dμ = Ae_delay*μ (sparse*dense) + head row correction + ce.
            mul!(dμ, Ae_delay, μ)
            # head row: dμ[r0] += At*μ[r0] + Bt*μ_del  + ct
            @views dμ[r0] .+= At * y0v .+ Bt * μ_del .+ ct

            # --- dP = Ae*P + P*Ae' + Q  -------------------------------------
            # Ae*P        = Ae_delay*P + H*P  (H*P nonzero only on rows r0)
            # P*Ae'       = P*Ae_delay' + P*H' (P*H' nonzero only on cols r0)
            # Q has only the r0×r0 block nonzero (= EgEg).
            mul!(dP, Ae_delay, P)                # dP  = Ae_delay * P
            mul!(dP, P, Ae_delayT, 1.0, 1.0)     # dP += P * Ae_delay'

            # P_row_del and P_col_del = weighted sum of the two grid rows/cols of P.
            @views P_row_del .= s1 .* P[rj, :] .+ s .* P[rjp, :]
            @views P_col_del .= s1 .* P[:, rj] .+ s .* P[:, rjp]

            # head_row = At * P[r0, :] + Bt * P_row_del   (n × N)
            @views mul!(head_row, At, P[r0, :])
            mul!(head_row, Bt, P_row_del, 1.0, 1.0)

            # head_col = P[:, r0] * At' + P_col_del * Bt' (N × n)
            @views mul!(head_col, P[:, r0], transpose(At))
            mul!(head_col, P_col_del, transpose(Bt), 1.0, 1.0)

            @views dP[r0, :] .+= head_row
            @views dP[:, r0] .+= head_col

            # Q contribution
            @views dP[r0, r0] .+= EgEg
        end

        return nothing
    end

    f_dde!(dy, y, h, p, t) = f!(dy, y, p, t)

    hist(p, s) = y0;

    prob = dde ? DDEProblem(f_dde!, y0, hist, tspan) : ODEProblem(f!, y0, tspan)

    meta = (Ae_delay=Ae_delay, ce=ce_delay, h=h, N=N, n=n, m=m, τ_max=τ_grid)

    return prob, meta
end

function parseFunction(f::Function, t, args...)
    try
        return f(t,args...)
    catch
        return f(t)
    end
end

function parseFunction(f::Number, t, args...)
    return f
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
