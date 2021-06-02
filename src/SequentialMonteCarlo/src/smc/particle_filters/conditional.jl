abstract type  AbstractConditionalParticleFilter <: AbstractParticleFilter end

"""
Extract the refernce trajectory from a conditional particle filter.
"""
reference(f::AbstractConditionalParticleFilter) = states(f)[1,:]

function simulate_initial!(f::AbstractConditionalParticleFilter, θ)
    model = _model(f)
    data = _data(f)
    n_particles = _n_particles(f)
    proposal = _proposal(f)
    X₁,_ = states(f, 1)
    logWₜ,_ = weights(f, 1)
    logWₜ[1] -= log_proposal_density(proposal, X₁[1], data, θ)
    @inbounds for n ∈ 2:n_particles
        logWₜ[n] -= simulate_proposal!(proposal, X₁[n], data, θ)
    end
end

function propagate!(f::AbstractConditionalParticleFilter, t, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    n_particles = _n_particles(f)
    Xₜ,Xₜ₋₁ = states(f, t)
    logWₜ,_ = weights(f, t)
    A = ancestors(f, t-1)
    logWₜ[1] -= log_proposal_density(proposal, Xₜ[1], Xₜ₋₁[1], t-1, data, θ)
    for i ∈ 2:n_particles
        # simulate_proposal!(proposal, Xₜ[i], Xₜ₋₁[A[i]], t-1, data, θ)
        logWₜ[i] -= simulate_proposal!(proposal, Xₜ[i], Xₜ₋₁[A[i]], t-1, data, θ)
    end
end

"""
Sample new reference trajectory and store it as the first particle at every iteration.
"""
function finalize!(f::AbstractConditionalParticleFilter, θ)
    W = normalized_weights(f)
    idx = sample_one_index(W) # sample new reference
    condition_on_particle!(f, idx) # condition on sampled reference.
end

struct ConditionalParticleFilter{T<:ParticleFilter} <: AbstractConditionalParticleFilter
    pf::T
    function ConditionalParticleFilter(pf::ParticleFilter)
        proposal = _proposal(pf)
        potential = _potential(pf)
        resampler = conditional(_resampler(pf))
        storage = _storage(pf)
        pf = ParticleFilter(proposal, potential, resampler, storage)
        return new{typeof(pf)}(pf)
    end
end
conditional(pf::ParticleFilter) = ConditionalParticleFilter(pf)
ConditionalParticleFilter(args...; kwargs...) = ConditionalParticleFilter(ParticleFilter(args...;kwargs...))

_pf(f::ConditionalParticleFilter) = f.pf

struct ConditionalParticleFilterWithAncestorSampling{T<:ParticleFilter} <: AbstractConditionalParticleFilter
    pf::T
    function ConditionalParticleFilterWithAncestorSampling(pf::ParticleFilter)
        proposal = _proposal(pf)
        potential = _potential(pf)
        resampler = conditional(_resampler(pf))
        storage = _storage(pf)
        pf = ParticleFilter(proposal, potential, resampler, storage)
        return new{typeof(pf)}(pf)
    end
end
_pf(f::ConditionalParticleFilterWithAncestorSampling) = f.pf
ConditionalParticleFilterWithAncestorSampling(args...; kwargs...) = ConditionalParticleFilterWithAncestorSampling(ParticleFilter(args...;kwargs...))


# Overload this if we want ancestor sampling.
function resample!(f::ConditionalParticleFilterWithAncestorSampling, t, θ)
    logWₜ, logWₜ₋₁ = weights(f, t)
    W = normalized_weights(f)
    A = ancestors(f, t-1)
    resample!(
        _resampler(f),
        A,
        W,
        logWₜ₋₁,
        logWₜ)
    weight_ancestors!(f, t, θ)
    A[1] = sample_one_index(normalized_weights(f))
end

"""
Computes ancestor weights for `states(f, t)[1]` and stores them in `normalized_weights(f)`.
"""
function weight_ancestors!(f, t, θ)
    model = _model(f)
    data = _data(f)
    Xₜ,Xₜ₋₁ = states(f, t) # the reference particle
    Xₜ = Xₜ[1]
    W = normalized_weights(f) # Borrow this to store weights
    _,logWₜ₋₁ = weights(f, t)
    if _potential(f) isa IdentityPotential # This should be compiled away
        for i in eachindex(Xₜ₋₁)
            W[i] = logWₜ₋₁[i] + log_transition_density(Xₜ, Xₜ₋₁[i], model, t-1, data, θ)
        end
    else
        _,logΨₜ₋₁ = potentials(f, t)
        for i in eachindex(Xₜ₋₁)
            W[i] = logWₜ₋₁[i] + log_transition_density(Xₜ, Xₜ₋₁[i], model, t-1, data, θ) - logΨₜ₋₁[i]
        end
    end
    expnormalize!(W)
end


"""
Sample one index P(i) = Wᵢ. Assumes `sum(W) == 1`.
"""
function sample_one_index(W)
    u = rand()
    csum = zero(Float64)
    N = length(W)
    @inbounds for i in 1:N
        csum += W[i]
        if u <= csum
            return i
        end
    end
    return N
end

"""
Condition on particle ending in X[idx, T] by moving the whole
trajectory to X[1, :].
"""
function condition_on_particle!(f::AbstractConditionalParticleFilter, idx::Integer)
    N,T = capacity(f)
    Xₜ,_ = states(f, T)
    copy!(Xₜ[1], Xₜ[idx])
    for t = T-1:-1:1
        A = ancestors(f, t)
        idx = A[idx]
        !(0 < idx <= N) && @error "ancerstor index out of range: A[?,$t] = $idx"

        Xₜ,Xₜ₋₁ = states(f, t)
        copy!(Xₜ[1], Xₜ[idx])
        # swap!(states(f, t)[1][idx], states(f, t)[1][1])
    end
end
