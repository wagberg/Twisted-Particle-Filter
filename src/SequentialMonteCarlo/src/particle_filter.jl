abstract type SSMStorage end

"""
Preallocate storage for particle filter.

"""
struct ParticleStorage{P <: Particle, T <: AFloat} <: SSMStorage
    X::Matrix{P}
    W::Matrix{T}
    A::Matrix{Int}
    V::Vector{T}
    wnorm::Vector{T}
    ref::Vector{T}
    filtered_index::Base.RefValue{Int}

    function ParticleStorage(::Type{P}, n_particles::Integer, t::Integer) where {P<:Particle}
        X = [P() for n in 1:n_particles, j in 1:t]
        W = zeros(Float64, n_particles, t)
        V = zeros(Float64, n_particles)
        wnorm = zeros(Float64, n_particles)
        A = zeros(typeof(n_particles), n_particles, t)
        ref = ones(typeof(n_particles), t)
        filtered_index = Ref(0)
        new{P, Float64}(X, W, A, V, wnorm, ref, filtered_index)
    end
end

function particle_count(storage::ParticleStorage)
    size(storage.X, 1)
end


function pf!(storage::ParticleStorage, model::SSM, data, θ;
        resampling::Resampling = MultinomialResampling(), conditional::Bool = false, n_particles::Int = particle_count(storage) )
    
    @assert n_particles <= particle_count(storage)
    T = length(data.y)
    X = view(storage.X, 1:n_particles, :);
    W = view(storage.W, 1:n_particles, :);
    A = view(storage.A, 1:n_particles, :);

    ll = 0

    for j in 1:n_particles
        @inbounds simulate_initial!(X[j,1], model, data, θ)
        @inbounds W[j,1] = log_observation_density(X[j,1], model, 1, data, θ)
    end

    logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    W[:, 1] .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t-1)
        resample!(a, W[:, t-1], resampling)
        
        for j in 1:n_particles
            simulate_transition!(X[j,t], X[a[j],t-1], model, t-1, data, θ)
        end

        for j in 1:n_particles
            @inbounds W[j,t] = log_observation_density(X[j,t], model, t, data, θ)
        end
        logΣexp = logsumexp(W[:, t])
        ll += logΣexp - log(n_particles)
        W[:, t] .= exp.(W[:, t] .- logΣexp)
    end
    ll;
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