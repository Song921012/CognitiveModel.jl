using CognitiveModel
using Test

@testset "CognitiveModel.jl" begin
    # Write your tests here.
    s0 = 1000
    v0=0
    e0=1
    i0=1
    ro=0
    βinit=0.3
    m0=0
    c0=1
    u0 = [s0, v0, e0, i0, ro, βinit, m0, c0]

    N = s0+v0+e0+i0+ro
    ν0=1/500
    n=1
    b=4
    ϵ=0.8
    σ=0.2
    γ=0.1
    k=1
    β0=0.3
    η=1.2
    ξ=0.3
    rm=0.5 
    rc=1.0
    α=2.0 
    θ=1.0
    c=4.0
    r=1
    p = [N, ν0, n, b, ϵ, σ, γ, k, β0, η, ξ, rm, rc, α, θ, c, r]
    Tmax=100.0
    tspan = (0, Tmax)
    prob = ODEProblem(cognitive!, u0, tspan, p)
    sol = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10)
    plot(sol)
end
