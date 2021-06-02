abstract type AbstractParticleFilter <: AbstractSMC end

"""
Retruns the underlying particle filter object.
"""
function _pf(::AbstractParticleFilter) end

_storage(f::AbstractParticleFilter) = _storage(_pf(f))
_resampler(f::AbstractParticleFilter) = _resampler(_pf(f))
_proposal(f::AbstractParticleFilter) = _proposal(_pf(f))
_potential(f::AbstractParticleFilter) = _potential(_pf(f))
capacity(f::AbstractParticleFilter,args...) =  capacity(_storage(f),args...)
proposal_type(f::AbstractParticleFilter) = proposal_type(_pf(f))
potential_type(f::AbstractParticleFilter) = potential_type(_pf(f))
resampler_type(f::AbstractParticleFilter) = resampler_type(_pf(f))

function simulate_initial!(f::AbstractParticleFilter, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    X₁,_ = states(f, 1)
    logW₁,_ = weights(f, 1)
    @inbounds for n ∈ eachindex(X₁)
        logW₁[n] -= simulate_proposal!(proposal, X₁[n], data, θ)
    end
end

function weight!(f::AbstractParticleFilter, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    potential = _potential(f)
    logW₁,_ = weights(f, 1)
    logΨ₁,_ = potentials(f, 1)
    X₁,_ = states(f, 1)
    @inbounds for i ∈ eachindex(X₁)
        logΨ₁[i] = log_potential(potential, X₁[i], 1, data, θ)
        log_obs_dens = log_observation_density(X₁[i], model, 1, data, θ)
        log_init_dens = log_initial_density(X₁[i], model, data, θ)
        # Already computed and subtracted in simulate_initial!.
        # log_proposal = log_proposal_density(proposal, X₁[i], data, θ)
        logW₁[i] += log_obs_dens + log_init_dens + logΨ₁[i]# - log_proposal
    end
end

function propagate!(f::AbstractParticleFilter, t, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    Xₜ,Xₜ₋₁ = states(f, t)
    logWₜ,_ = weights(f, t)
    A = ancestors(f, t-1)
    @inbounds for i in eachindex(Xₜ)
        logWₜ[i] -= simulate_proposal!(proposal, Xₜ[i], Xₜ₋₁[A[i]], t-1, data, θ)
    end
end

function weight!(f::AbstractParticleFilter, t, θ)
    model = _model(f)
    data = _data(f)
    proposal = _proposal(f)
    potential = _potential(f)
    A = ancestors(f, t-1)
    logWₜ,_ = weights(f, t)
    logΨₜ,logΨₜ₋₁ = potentials(f, t)
    Xₜ,Xₜ₋₁ = states(f, t)
    @inbounds for i ∈ eachindex(logWₜ)
        logΨₜ[i] = log_potential(potential, Xₜ[i], t, data, θ)
        log_obs_dens = log_observation_density(Xₜ[i], model, t, data, θ)
        log_trans_dens = log_transition_density(Xₜ[i], Xₜ₋₁[A[i]], model, t-1, data, θ)
        # Already subtracted in propagate!
        # log_proposal = log_proposal_density(proposal, Xₜ[i], Xₜ₋₁[A[i]], t-1, data, θ)
        logWₜ[i] += log_obs_dens + log_trans_dens + logΨₜ[i] - logΨₜ₋₁[A[i]]# - log_proposal
    end
end

###############
###  SHOW  ####
###############
# function Base.show(io::IO, pf::AbstractParticleFilter)
#     println(io, name(pf))
#     println(io, "  Model: $(_model(pf))")
#     println(io, "  Particles: $(_n_particles(pf))/$(particle_count(pf))")
#     println(io, "  Datalength: $(length(get_data(pf).y))")
#     println(io, "  Proposal: $(_proposal(pf))")
#     println(io, "  Potential: $(_potential(pf))")
#     print(io, "  Resampler: $(_resampler(pf))")
# end

name(::AbstractParticleFilter) = "Particle filter"

include("particle_filter.jl")
include("bootstrap_particle_filter.jl")
# include("twisted_particle_filter.jl")
include("conditional.jl")
