abstract type SSMStorage end

"""
Preallocate storage for particle filter.

"""
struct ParticleStorage{S <: Particle,T <: AFloat} <: SSMStorage
    X::Matrix{S} # Particles for each time step
    W::Matrix{T} # Unnormalized log weights
    A::Matrix{Int} # Ancestors X[i, t] ~ p(. | X[A[i,t-1], t-1])
    P::Matrix{T} # log potentials ψₜ(X[i, t])
    wnorm::Vector{T} # Used to normalize log weights
    V::Vector{T}
    ref::Vector{T}
    filtered_index::Base.RefValue{Int}

    function ParticleStorage(::Type{S}, n_particles::Integer, t::Integer) where {S <: Particle}
        X = [S() for n in 1:n_particles, j in 1:t]
        W = zeros(Float64, n_particles, t)
        P = zeros(Float64, n_particles, t)
        V = zeros(Float64, n_particles)
        wnorm = zeros(Float64, n_particles)
        A = zeros(typeof(n_particles), n_particles, t)
        ref = ones(typeof(n_particles), t)
        filtered_index = Ref(0)
        new{S,Float64}(X, W, A, P, wnorm, V, ref, filtered_index)
    end
end

ParticleStorage(model::SSM, n_particles, t) = ParticleStorage(particletype(model), n_particles, t)

function particle_count(storage::ParticleStorage)
    size(storage.X, 1)
end

"""
Bootstrap particle filter.
Uses transition distribution as proposal.
Never evaluates transition density. Only requires observation density.
"""
function bpf!(storage::ParticleStorage, model::SSM, data, θ;
        resampling::Resampling=MultinomialResampling(), conditional::Bool=false, n_particles::Int=particle_count(storage) )
    
    @assert n_particles <= particle_count(storage)
    T = length(data.y)
    X = view(storage.X, 1:n_particles, :);
    W = view(storage.W, 1:n_particles, :);
    A = view(storage.A, 1:n_particles, :);
    wnorm = view(storage.wnorm, 1:n_particles);
    # ref = view(storage.ref, 1:n_particles)

    ll = 0

    start = conditional ? 2 : 1
    for j in start:n_particles
        @inbounds simulate_initial!(X[j,1], model, data, θ)
    end

    for j in 1:n_particles
        @inbounds W[j,1] = log_observation_density(X[j,1], model, 1, data, θ)
    end

    @views logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    # W[:, 1] .= exp.(W[:, 1] .- logΣexp)
    wnorm .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t - 1)
        # resample!(a, W[:, t - 1], resampling, conditional)
        resample!(a, wnorm, resampling, conditional)
        
        for j in start:n_particles
            simulate_transition!(X[j,t], X[a[j],t - 1], model, t - 1, data, θ)
        end

        for j in 1:n_particles
            @inbounds W[j,t] = log_observation_density(X[j,t], model, t, data, θ)
        end
        @views logΣexp = logsumexp(W[:, t])
        ll += logΣexp - log(n_particles)
        # W[:, t] .= exp.(W[:, t] .- logΣexp)
        @views wnorm .= exp.(W[:, t] .- logΣexp)
    end
    ll;
end


function log_potential(p::FloatParticle, model::SSM, t, data, θ)
    # p(y_{t+1:T} | x_t) = p(x_t | y_{1:T}) * p(y_{1:T}) / (p(x_{t} | y_{1:t}) * p(y_{1:t}))
    (logpdf(MvNormal(data.ks.smooth_mean[t], data.ks.smooth_Sigma[t]), p.x)
        + data.ks.log_likelihood[end]
        - logpdf(MvNormal(data.ks.filter_mean[t], data.ks.filter_Sigma[t]), p.x)
        - data.ks.log_likelihood[t])
end

function simulate_proposal!(pnext::FloatParticle, pcurr::FloatParticle, model, t, data, θ)
    pnext.x .= rand(MvNormal(data.ks.smooth_mean[t], data.ks.smooth_Sigma[t]))
end

simulate_proposal!(p::FloatParticle, model, data, θ) = simulate_proposal!(p, p, model, 1, data, θ)

function log_proposal_density(pnext::FloatParticle, pcurr::FloatParticle, model, t, data, θ)
    logpdf(MvNormal(data.ks.smooth_mean[t], data.ks.smooth_Sigma[t]), pnext.x)
end

log_proposal_density(p::FloatParticle, model, data, θ) = log_proposal_density(p, p, model, 1, data, θ)

function tpf!(storage::ParticleStorage, model::SSM, data, θ;
    resampling::Resampling=MultinomialResampling(), conditional::Bool=false, n_particles::Int=particle_count(storage) )

    @assert n_particles <= particle_count(storage)
    T = length(data.y)
    X = view(storage.X, 1:n_particles, :);
    W = view(storage.W, 1:n_particles, :);
    A = view(storage.A, 1:n_particles, :);
    P = view(storage.P, 1:n_particles, :);
    wnorm = view(storage.wnorm, 1:n_particles);

    ll = 0

    # Simulate initial state
    for j in 1:n_particles
        # @inbounds simulate_initial!(X[j,1], model, data, θ)
        @inbounds simulate_proposal!(X[j, 1], model, data, θ)
    end

    # Compute weights
    for j in 1:n_particles
        @inbounds P[j, 1] = log_potential(X[j, 1], model, 1, data, θ)
        @inbounds W[j, 1] = (log_observation_density(X[j, 1], model, 1, data, θ)
            + log_initial_density(X[j, 1], model, data, θ)
            + P[j, 1]
            - log_proposal_density(X[j, 1], model, data, θ))
    end

    logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    @views wnorm .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t - 1)
        resample!(a, wnorm, resampling)
        
        for j in 1:n_particles
            # @inbounds simulate_transition!(X[j, t], X[a[j], t - 1], model, t - 1, data, θ)
            @inbounds simulate_proposal!(X[j, t], X[a[j], t-1], model, t, data, θ)
        end

        for j in 1:n_particles
            @inbounds P[j, t] = log_potential(X[j, t], model, t, data, θ)
            @inbounds W[j, t] = (
                   log_transition_density(X[j, t], X[a[j], t-1], model, t-1, data, θ)
                 + log_observation_density(X[j, t], model, t, data, θ)
                 + P[j, t]
                 - P[a[j], t-1]
                 - log_proposal_density(X[j, t], X[a[j], t-1], model, t-1, data, θ))
        end
        @views logΣexp = logsumexp(W[:, t])
        ll += logΣexp - log(n_particles)
        @views wnorm .= exp.(W[:, t] .- logΣexp)
    end
    ll;
end

"""
Condition on particle ending in X[idx, T] by moving the whole
trajectory to X[1, :].
"""
function condition_on_particle!(ps::ParticleStorage, idx::Integer)
    T = size(ps.X, 2)
    swap!(ps.X[1, T], ps.X[idx, T])

    for t = T-1:-1:1
        idx = ps.A[idx, t]
        swap!(ps.X[1, t], ps.X[idx, t])
    end
end

"""
Normalise a vector of weight logarithms, `log_weights`, in place.
After normalisation, the weights are in the linear scale.
Additionally, the logarithm of the linear scale mean weight is returned.
"""
# @inline function normalise_logweights!(log_weights::AVec{<: Real})
#   m = maximum(log_weights);
#   if isapprox(m, -Inf) # To avoid NaN in case that all values are -Inf.
#     log_weights .= zero(eltype(log_weights));
#     return -Inf;
#   end
#   log_weights .= exp.(log_weights .- m);
#   log_mean_weight = m + log(mean(log_weights));
#   normalize!(log_weights, 1);
#   log_mean_weight;
# end