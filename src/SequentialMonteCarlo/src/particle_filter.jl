abstract type SSMStorage end

"""
Preallocate storage for particle filter.

"""
struct ParticleStorage{P <: Particle,T <: AFloat} <: SSMStorage
    X::Matrix{P}
    Xref::Vector{P}
    W::Matrix{T}
    A::Matrix{Int}
    V::Vector{T}
    wnorm::Vector{T}
    ref::Vector{Int}  # Vector of indices for the reference trajectory, 
    filtered_index::Base.RefValue{Int}

    function ParticleStorage(::Type{P}, n_particles::Integer, t::Integer) where {P <: Particle}
        X = [P() for n in 1:n_particles, j in 1:t]
        Xref = [P() for j in 1:t]
        W = zeros(Float64, n_particles, t)
        V = zeros(Float64, n_particles)
        wnorm = zeros(Float64, n_particles)
        A = zeros(typeof(n_particles), n_particles, t)
        ref = zeros(typeof(n_particles), t)
        filtered_index = Ref(0)
        new{P,Float64}(X, Xref, W, A, V, wnorm, ref, filtered_index)      
    end
end

ParticleStorage(model::SSM, n_particles, t) = ParticleStorage(particletype(model), n_particles, t)

function particle_count(storage::ParticleStorage)
    size(storage.X, 1)
end

function sample_ref_ancestor_final(w::AVec{Float64}, rng::AbstractRNG = Random.GLOBAL_RNG)
    u = rand(rng)
    csum = zero(Float64)
    N = length(w)
    for i in 1:N
        csum += @inbounds w[i]
        if u <= csum
            return i
        end
    end
    N   # return N if loop terminates
end

function generate_trajectory(A::AMat{Int}, X::AMat{<:Particle}, ref::AVec{Int}, finalInd::Integer) # Add Xref::AVec{<:Particle}, as argumet if want to store Xref separately
    # Change X[1,t] to Xref[t] if want to store reference separately
    
    T = size(X, 2)
    
    ref[T] = finalInd  # Save sampled trajectory on last place in ref
    X[1,T] = X[finalInd,T]
    for t in (T-1):-1:1
        ref[t] = A[ref[t+1],t]
        X[1,t] = X[ref[t],t]
    end
    X[1,:]
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
    ref = view(storage.ref, 1:T)
    # Xref = view(storage.Xref,1:T)

    ll = 0

    start = conditional ? 2 : 1
    for j in start:n_particles
        @inbounds simulate_initial!(X[j,1], model, data, θ)
    end

    for j in 1:n_particles
        @inbounds W[j,1] = log_observation_density(X[j,1], model, 1, data, θ)
    end

    logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    W[:, 1] .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t - 1)
        resample!(a, W[:, t - 1], resampling, conditional)

        for j in start:n_particles
            simulate_transition!(X[j,t], X[a[j],t - 1], model, t - 1, data, θ)
        end

        for j in 1:n_particles
            @inbounds W[j,t] = log_observation_density(X[j,t], model, t, data, θ)
        end
        logΣexp = logsumexp(W[:, t])
        ll += logΣexp - log(n_particles)
        W[:, t] .= exp.(W[:, t] .- logΣexp)
    end

    if conditional
        finalInd = sample_ref_ancestor_final(W[:, T]);
        Xref = generate_trajectory(A, X, ref, finalInd);  # Add Xref here if want to store Xref separately between runs
        return Xref;
    end
    
    ll;
end


function log_potential(p::FloatParticle, model::SSM, t, data, θ)
    # p(y_{t+1:T} | x_t) = p(x_t | y_{1:T}) * p(y_{1:T}) / (p(x_{t} | y_{1:t}) * p(y_{1:t}))
    logpdf(MvNormal(data.ks.smooth_mean[t], data.ks.smooth_Sigma[t]), p.x) + 
        data.ks.log_likelihood[end] - 
        logpdf(MvNormal(data.ks.filter_mean[t], data.ks.filter_Sigma[t]), p.x) -
        data.ks.log_likelihood[t]
end


function tpf!(storage::ParticleStorage, model::SSM, data, θ;
    resampling::Resampling=MultinomialResampling(), conditional::Bool=false, n_particles::Int=particle_count(storage) )

    @assert n_particles <= particle_count(storage)
    T = length(data.y)
    X = view(storage.X, 1:n_particles, :);
    W = view(storage.W, 1:n_particles, :);
    A = view(storage.A, 1:n_particles, :);

    ll = 0

    for j in 1:n_particles
        @inbounds simulate_initial!(X[j,1], model, data, θ)
        @inbounds W[j,1] = log_observation_density(X[j,1], model, 1, data, θ) + log_potential(X[j,1], model, 1, data, θ)
    end

    logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    W[:, 1] .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t - 1)
        resample!(a, W[:, t - 1], resampling)
        
        for j in 1:n_particles
            simulate_transition!(X[j,t], X[a[j],t - 1], model, t - 1, data, θ)
        end

        for j in 1:n_particles
            @inbounds W[j,t] = log_observation_density(X[j,t], model, t, data, θ) + 
                log_potential(X[j,t], model, t, data, θ) -
                log_potential(X[a[j],t - 1], model, t - 1, data, θ)
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