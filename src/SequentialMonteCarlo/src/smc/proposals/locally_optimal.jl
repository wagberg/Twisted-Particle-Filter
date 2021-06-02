"""
Approximate (or exact for additive Guassian transition noise and linear Gaussian obsevations)
locally optimal proposal.
"""
struct LocallyOptimalProposal{MT<:AbstractSSM,FT<:AbstractGaussianFilter} <: AbstractProposal
    model::MT
    f::FT
end
function LocallyOptimalProposal(model::AbstractSSM{<:FloatParticle{N,T}}, f::Type{<:AbstractGaussianFilter} = KalmanFilter) where {N,T}
    storage = EmptyGaussianStorage()
    LocallyOptimalProposal(model, f(storage))
end

function log_proposal_density(p::LocallyOptimalProposal, x₁, data, θ)
    pd = MvNormal(initial_mean(p.model, data, θ), initial_covariance(p.model, data, θ))
    _,fd = observe(pd, p.f, p.model, 1, data, θ)
    return logpdf(fd, x₁.x)
end

function log_proposal_density(p::LocallyOptimalProposal, xₜ₊₁::FloatParticle{N,T}, xₜ, t, data, θ) where {N,T}
    x = toSVector(xₜ)
    Σ = Cholesky(zero(SMatrix{N,N,T}), 'U', 0)
    Σ⁺ = PDMat(zero(SMatrix{N,N,T}),Σ)
    id = MvNormal(x, Σ⁺)
    pd = predict(id, p.f, p.model, t, data, θ)
    _,fd = observe(pd, p.f, p.model, t+1, data, θ)
    return logpdf(fd, xₜ₊₁.x)
end

function simulate_proposal!(p::LocallyOptimalProposal, x₁, data, θ)
    pd = MvNormal(initial_mean(p.model, data, θ), initial_covariance(p.model, data, θ))
    _,fd = observe(pd, p.f, p.model, 1, data, θ)
    rand!(fd, x₁.x)
    return logpdf(fd, x₁.x)
end

function simulate_proposal!(p::LocallyOptimalProposal, xₜ₊₁::FloatParticle{N,T}, xₜ, t, data, θ) where {N,T}
    x = toSVector(xₜ)
    Σ = Cholesky(zero(SMatrix{N,N,T}), 'U', 0)
    Σ⁺ = PDMat(zero(SMatrix{N,N,T}),Σ)
    id = MvNormal(x, Σ⁺)
    pd = predict(id, p.f, p.model, t, data, θ)
    _,fd = observe(pd, p.f, p.model, t+1, data, θ)
    rand!(fd, xₜ₊₁.x)
    return logpdf(fd, xₜ₊₁.x)
end
