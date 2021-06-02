"""
    ParticleFilter <: AbstractParticleFilter

General particle filter.

"""
struct ParticleFilter{PrT,PoT,RT,ST} <: AbstractParticleFilter
    proposal::PrT
    potential::PoT
    resampler::RT
    storage::ST
end

"""
ParticleFilter(model, n_particles, data, proposal, potential, resampler)
"""
function ParticleFilter(model::AbstractSSM, n_particles::Int, data;
        proposal::AbstractProposal = BootstrapProposal(model),
        potential::AbstractPotential = IdentityPotential(),
        resampler::AbstractResampler=ResampleWithESSThreshold(),
        storage::Type{<:AbstractParticleStorage} = FullParticleStorage)
    s = storage(model, n_particles, data)
    return ParticleFilter(proposal, potential, resampler, s)
end

function ParticleFilter(pf::ParticleFilter;
        proposal=_proposal(pf),
        potential=_potential(pf),
        resampler=_resampler(pf))
    return ParticleFilter(proposal, potential, resampler, _storage(pf))
end


potential_type(::ParticleFilter{<:Any, PoT}) where PoT = PoT
proposal_type(::ParticleFilter{PrT}) where PrT = PrT
resampler_type(::ParticleFilter{<:Any,<:Any,RT}) where RT = RT

_pf(f::ParticleFilter) = f
_storage(f::ParticleFilter) = f.storage
_resampler(f::ParticleFilter) = f.resampler
_proposal(f::ParticleFilter) = f.proposal
_potential(f::ParticleFilter) = f.potential

"""
Special simulate_initial! for Bootstrap proposal to not evaluate initial density
"""
function simulate_initial!(f::ParticleFilter{<:BootstrapProposal}, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    X₁,_ = states(f, 1)
    @inbounds for n ∈ eachindex(X₁)
        simulate_proposal!(proposal, X₁[n], data, θ)
    end
end

"""
Special weighting for bootstrap proposal to not evaluate transition density, since it might
not exist.
"""
function weight!(f::ParticleFilter{<:BootstrapProposal}, θ)
    s = _storage(f)
    n_particles = _n_particles(s)
    model = _model(f)
    data = _data(f)
    potential = _potential(f)
    logW₁,_ = weights(s, 1)
    logΨ₁,_ = potentials(s, 1)
    X₁,_ = states(s, 1)
    for i ∈ eachindex(X₁)
        logΨ₁[i] = log_potential(potential, X₁[i], 1, data, θ)
        log_obs_dens = log_observation_density(X₁[i], model, 1, data, θ)
        logW₁[i] += log_obs_dens + logΨ₁[i]
    end
end


"""
Special propagate for Bottstrap proposal to not evaluate transition density.
"""
function propagate!(f::ParticleFilter{<:BootstrapProposal}, t, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    Xₜ,Xₜ₋₁ = states(f, t)
    logWₜ,_ = weights(f, t)
    A = ancestors(f, t-1)
    for i in eachindex(Xₜ)
        simulate_proposal!(proposal, Xₜ[i], Xₜ₋₁[A[i]], t-1, data, θ)
    end
end

"""
Special weighting for bootstrap proposal to not evaluate transition density, since it might
not exist.
"""
function weight!(f::ParticleFilter{<:BootstrapProposal}, t, θ)
    model = _model(f)
    data = _data(f)
    potential = _potential(f)
    A = ancestors(f, t-1)
    logWₜ,_ = weights(f, t)
    logΨₜ,logΨₜ₋₁ = potentials(f, t)
    Xₜ,Xₜ₋₁ = states(f, t)
    for i ∈ eachindex(logWₜ)
        logΨₜ[i] = log_potential(potential, Xₜ[i], t, data, θ)
        log_obs_dens = log_observation_density(Xₜ[i], model, t, data, θ)
        logWₜ[i] += log_obs_dens + logΨₜ[i] - logΨₜ₋₁[A[i]]
    end
end


function resample!(f::ParticleFilter{<:Any,<:Any,<:Any,<:SlimParticleStorage}, t, θ)
    logWₜ, logWₜ₋₁ = weights(f, t)
    A = ancestors(f, t-1)
    resample!(
        _resampler(f),
        A,
        normalized_weights(f),
        logWₜ₋₁,
        logWₜ)
end
