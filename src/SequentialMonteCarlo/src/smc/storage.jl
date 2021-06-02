abstract type AbstractParticleStorage end

"""
Returns the total number of pre-allocated prtialces.
"""
capacity(s::AbstractParticleStorage) = size(all_states(s))
capacity(s::AbstractParticleStorage, dim) = size(all_states(s), dim)
"""
Returns the number of pre-allocated particles per time step.
"""
particle_count(s::AbstractParticleStorage) = capacity(s, 1)
"""
Returns the `n_particles` first normalized weights.
"""
normalized_weights(s::AbstractParticleStorage) = view(all_normalized_weights(s), 1:_n_particles(s))
"""
Returns the first `n_particles` particles for all timesteps.
"""
states(s::AbstractParticleStorage) = view(all_states(s), 1:_n_particles(s), :)
"""
Returns the first `n_particles` particles at time step `t`.
"""
states(s::AbstractParticleStorage, t) = (view(states(s), :,  t), view(states(s), :,  max(t-1,1)))
"""
Returns the first `n_particles` weights for all time steps.
"""
weights(s::AbstractParticleStorage) = view(all_weights(s), 1:_n_particles(s), :)
"""
Returns the first `n_particles weights for time step `t`.
"""
weights(s::AbstractParticleStorage, t) = (view(weights(s), :, t), view(weights(s), :, max(t-1,1)))
"""
Returns the first `n_particles` potenials for all time steps.
"""
potentials(s::AbstractParticleStorage) = view(all_potentials(s), 1:_n_particles(s), :)
"""
Returns the first `n_particles` potenials for time step `t`.
"""
potentials(s::AbstractParticleStorage, t) = (view(potentials(s), :, t), view(potentials(s), :, max(t-1, 1)))
"""
Returns the first `n_particles` ancestors for all time steps.
"""
ancestors(s::AbstractParticleStorage) = view(all_ancestors(s), 1:_n_particles(s), :)
"""
Returns the first `n_particles` ancestors for time steps `t`.
"""
ancestors(s::AbstractParticleStorage, t) = view(ancestors(s), :, t)

"""
Normalize log weights at time t and save exp weights in normlaized weights.
Returns the logΣexp as an esitmate of Zₜ/Zₜ₋₁.
"""
function normalize!(s::AbstractParticleStorage, t)
    ll = logsumexp!(weights(s, t)[1], normalized_weights(s))
    if t > 1
        s.ll[t] = ll + s.ll[t-1]
    else
        s.ll[t] = ll
    end
end

struct FullParticleStorage{MT,DT,XT,WT,wT,AT,ΨT,LT} <: AbstractParticleStorage
    model::MT
    data::DT  # Data connected with model
    X::XT # Particle trajectories
    logW::WT # log weights
    W::wT # weights
    A::AT # Ancestor indices
    logΨ::ΨT # log potentials. Need all potentials to do blocking.
    ll::LT # log likelihood estimates
    n_particles::Base.RefValue{Int} # Number of particles currently used (may differ from number allocated)
end
function FullParticleStorage(model, n_particles, data)
    P = particletype(model)
    T = length(data.y)
    X = [P() for n in 1:n_particles, j in 1:T]
    logW = zeros(Float64, n_particles, T+1)
    W = zeros(Float64, n_particles)
    A = zeros(Int, n_particles, T)
    logΨ = zeros(Float64, n_particles, T)
    ll = zeros(Float64, T)
    return FullParticleStorage(model, data, X, logW, W, A, logΨ, ll, Ref(n_particles))
end

_model(s::FullParticleStorage) = s.model
_data(s::FullParticleStorage) = s.data
_ll(s::FullParticleStorage) = s.ll
_ll(s::FullParticleStorage, t) = _ll(s)[t]
_n_particles(s::FullParticleStorage) = s.n_particles[]
_n_particles!(s::FullParticleStorage, n) = (@assert (n < capacity(s, 1)); s.n_particles[] = n)
all_normalized_weights(s::FullParticleStorage) = s.W
all_states(s::FullParticleStorage) = s.X
all_weights(s::FullParticleStorage) = s.logW
all_potentials(s::FullParticleStorage) = s.logΨ
all_ancestors(s::FullParticleStorage) = s.A

struct SlimParticleStorage{MT,DT,XT,WT,wT,AT,ΨT,LT} <: AbstractParticleStorage
    model::MT
    data::DT  # Data connected with model
    X::XT # Particle trajectories
    logW::WT # log weights
    W::wT # weights
    A::AT # Ancestor indices
    logΨ::ΨT # log potentials. Need all potentials to do blocking.
    ll::LT # log likelihood estimates
    n_particles::Base.RefValue{Int} # Number of particles
end
function SlimParticleStorage(model, n_particles, data)
    P = particletype(model)
    T = length(data.y)
    X = [P() for n in 1:n_particles, j in 1:2]
    logW = zeros(Float64, n_particles, 2)
    W = zeros(Float64, n_particles)
    A = zeros(Int, n_particles)
    logΨ = zeros(Float64, n_particles, 2)
    ll = zeros(Float64, T)
    return SlimParticleStorage(model, data, X, logW, W, A, logΨ, ll, Ref(n_particles))
end

_model(s::SlimParticleStorage) = s.model
_data(s::SlimParticleStorage) = s.data
_ll(s::SlimParticleStorage) = s.ll
_ll(s::SlimParticleStorage, t) = _ll(s)[t]
_n_particles(s::SlimParticleStorage) = s.n_particles[]
_n_particles!(s::SlimParticleStorage, n) = (@assert (n < capacity(s, 1)); s.n_particles[] = n)
all_normalized_weights(s::SlimParticleStorage) = s.W
all_states(s::SlimParticleStorage) = s.X
all_weights(s::SlimParticleStorage) = s.logW
all_potentials(s::SlimParticleStorage) = s.logΨ
all_ancestors(s::SlimParticleStorage) = s.A
ancestors(s::SlimParticleStorage, t) = view(ancestors(s), :)
potentials(s::SlimParticleStorage, t) = (view(potentials(s), :, min(2,t)), view(potentials(s), :, 1))
weights(s::SlimParticleStorage, t) = (view(weights(s), :, min(2,t)), view(weights(s), :, 1))
states(s::SlimParticleStorage, t) = (view(states(s), :,  min(2,t)), view(states(s), :,  1))


function normalize!(s::SlimParticleStorage, t)
    W = normalized_weights(s)
    if t > 1
        logWₜ,logWₜ₋₁ = weights(s, t)

        ll = logsumexp!(logWₜ, W)
        s.ll[t] = ll + s.ll[t-1]
        logΨₜ,logΨₜ₋₁ = potentials(s, t)
        # move states, log weights and log potentials to first column.
        Xₜ,Xₜ₋₁ = states(s, t)
        @turbo @. logWₜ₋₁ = logWₜ
        @turbo @. logΨₜ₋₁ = logΨₜ
        for i in eachindex(Xₜ)
            copy!(Xₜ₋₁[i], Xₜ[i])
        end
    else
        # States, weights and potentials already in first column. No need to move.
        logW₁,_ = weights(s, 1)
        ll = logsumexp!(logW₁, W)
        s.ll[1] = ll
    end
end
