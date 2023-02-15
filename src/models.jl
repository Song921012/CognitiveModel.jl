function cognitive!(du, u, p, t)
    S, V, E, I, R, β, M, C = u
    N, ν0, n, b, ϵ, σ, γ, k, β0, η, ξ, rm, rc, α, θ, c, r = p
    ν = ν0 * (1 - M^n / (b^n + M^n))
    dS = -β * S * I / N - ν * S
    dV = -(1 - ϵ) * β * V * I / N + ν * S
    dE = β * S * I / N + (1 - ϵ) * β * V * I / N - σ * E
    dI = σ * E - γ * I
    dR = γ * I
    dβ = k * β * (β0 * η - β) * (M - C) + ξ * (1 - β / β0)
    dM = rm - (α + η * C^r / (c^r + C^r)) * M
    dC = rc - θ * C
    du .= [dS, dV, dE, dI, dR, dβ, dM, dC]
end

