struct TwistedRTSParticleFilter{PT, RT} <: AbstractParticleFilter
    pf::PT
    rts::RT
end

function TwistedRTSParticleFilter(model, n_particles, data, resampler=SystematicResampler())
    rts = RTSSmoother(model, data)
    potential = RTSPotential(rts)
    proposal = RTSProposal(rts)
    pf = ParticleFilter(model, n_particles, data, proposal, potential, resampler)
    return TwistedRTSParticleFilter(pf, rts)
end

particle_filter(pf::TwistedRTSParticleFilter) = pf.pf
name(::TwistedRTSParticleFilter) = "Twisted RTS particle filter"

function (pf::TwistedRTSParticleFilter)(θ, args...; kwargs...)
    pf.rts(θ)
    init!(get_proposal(pf), θ)
    return run_filter!(pf, θ, args...; kwargs...)
end

function weight!(s::ParticleStorage, f::AbstractParticleFilter, t, θ)
    data = _data(s)
    model = _model(s)
    potential = _potential(f)
    proposal = _proposal(f)

    W = weights(s, t)
    Ψ = potentials(s, t)
    Xₜ = states(s, t)
    Xₜ₋₁ = states(s, t-1)
    @inbounds for i in 1:_n_particles(s)
        Ψ[i] = log_potential(potential, X[i], t, data, θ)
        log_obs_dens = log_observation_density(X[i], model, t, data, θ)
        log_init_dens = log_transition_density(Xₜ[i], Xₜ₋₁[i], model, t, data, θ)
        log_proposal = log_proposal_density(proposal, Xₜ[i], Xₜ₋₁[i], t, data, θ)
        W[i] += log_obs_dens + log_init_dens + Ψ[i] - log_proposal
    end
end
