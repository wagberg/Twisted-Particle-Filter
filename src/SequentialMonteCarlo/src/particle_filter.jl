abstract type SSMStorage end

"""
Preallocate storage for particle filter.

"""
struct ParticleStorage{P <: Particle,T <: AFloat} <: SSMStorage
    X::Matrix{P}
    Xref::Vector{P}
    Ψ::Matrix{T}
    W::Matrix{T}
    A::Matrix{Int}
    V::Vector{T}
    wnorm::Vector{T}
    wancestor::Vector{T}
    ref::Vector{Int}  # Vector of indices for the reference trajectory, 
    filtered_index::Base.RefValue{Int}

    function ParticleStorage(::Type{P}, n_particles::Integer, t::Integer) where {P <: Particle}
        X = [P() for n in 1:n_particles, j in 1:t]
        Xref = [P() for j in 1:t]
        W = zeros(Float64, n_particles, t)
        Ψ = zeros(Float64, n_particles, t)
        V = zeros(Float64, n_particles)
        wnorm = zeros(Float64, n_particles)
        wancestor = zeros(Float64, n_particles) # Ancestor weight
        A = zeros(typeof(n_particles), n_particles, t)
        ref = zeros(typeof(n_particles), t)
        filtered_index = Ref(0)
        new{P,Float64}(X, Xref, Ψ, W, A, V, wnorm, wancestor, ref, filtered_index)
    end
end
ParticleStorage(model::SSM, n_particles, t) = ParticleStorage(particletype(model), n_particles, t)


# """
# Pre-allocated storage for conditional particle filter (CPF)
# """
# struct CPFstorage{P <: Particle}
#     X::Vector{Vector{P}}
#     function CPFstorage(::Type{P}, time_steps::Integer, n_samples::Integer) where P<:Particle
#         X = [[P() for t in 1:time_steps] for n in 1:n_samples]
#         new{P}(X)
#     end
# end
# CPFstorage(model::SSM, time_steps, n_samples) = CPFstorage(particletype(model), time_steps, n_samples)


function particle_count(storage::ParticleStorage)
    size(storage.X, 1)
end

function sample_one_index(w::AVec{Float64}, rng::AbstractRNG = Random.GLOBAL_RNG)
    # TODO: Add check that w is normalized 
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
    X[1,T].x .= X[finalInd,T].x
    for t in (T-1):-1:1
        ref[t] = A[ref[t+1],t]
        X[1,t].x .= X[ref[t],t].x
    end
    #X[1,:]
end


#function generate_trajectory(A::AMat{Int}, X::AMat{<:Particle}, ref::AVec{Int}, finalInd::Integer) # Add Xref::AVec{<:Particle}, as argumet if want to store Xref separately
#    # Change X[1,t] to Xref[t] if want to store reference separately
#    println(typeof(X))
#    T = size(X, 2)
#    
#    ref[T] = finalInd  # Save sampled trajectory on last place in ref
#    X[1,T] = X[finalInd,T]
#    for t in (T-1):-1:1
#        ref[t] = A[ref[t+1],t]
#        X[1,t] = X[ref[t],t]
#    end
#    #X[1,:]
#end

function compute_ancestor_weights(Xi::AVec{<:Particle}, Xref::Particle, W::AVec{Float64}, wancestor::AVec{Float64}, model::SSM, t::Integer, data, θ) # (anc, xⁱₜ₋₁, xₜʳ,wⁱₜ₋₁ )
    n_particles = length(W)
    for i in 1:n_particles
        wancestor[i] = log(W[i]) + log_transition_density(Xref, Xi[i], model, t, data, θ)  # For SSM, no twisting
    end
    logΣexp = logsumexp(wancestor)
    wancestor .= exp.(wancestor .- logΣexp)
    #return wancestor
end

"""
Compute ancestor weights bpf
"""
function ancestor_weights!(wnorm, pcurr, pprev, model, t, data, θ)
    for i in eachindex(pprev)
        wnorm[i] = log(wnorm[i]) + log_transition_density(pcurr, pprev[i], model, t, data, θ) 
    end
    logΣexp = logsumexp(wnorm)
    wnorm .= exp.(wnorm .- logΣexp)
    nothing
end

"""
Compute ancestor weights tpf
"""
function ancestor_weights!(wnorm, pcurr, pprev, model, t, data, θ, Ψ::AVec{Float64})
    for i in eachindex(pprev)
        wnorm[i] = log(wnorm[i]) + log_transition_density(pcurr, pprev[i], model, t, data, θ) - Ψ[i]
    end
    logΣexp = logsumexp(wnorm)
    wnorm .= exp.(wnorm .- logΣexp)
    nothing
end


"""
Bootstrap particle filter.
Uses transition distribution as proposal.
Never evaluates transition density. Only requires observation density.

Arguments:
* `storage`: Preallocated storage for particle filter
* `resampling`: Type of resampling
* `contitional`: Run conditional particle filter. One of `:no`, `:yes`, `:as`, where :as uses ancestor sampling
* `n_particles`: Number of particles
"""
function bpf!(storage::ParticleStorage, model::SSM, data, θ;
        resampling::Resampling=MultinomialResampling(),
        conditional::Symbol=:no,
        ancestorsampling::Bool=false,
        n_particles::Int=particle_count(storage) )
    
    @assert n_particles <= particle_count(storage)
    T = length(data.y)
    X = view(storage.X, 1:n_particles, :);
    W = view(storage.W, 1:n_particles, :);
    A = view(storage.A, 1:n_particles, :);
    wnorm = view(storage.wnorm, 1:n_particles);
    wancestor = view(storage.wancestor, 1:n_particles);
    ref = view(storage.ref, 1:T)
    # Xref = view(storage.Xref,1:T)

    ll = 0

    start = conditional == :no ? 1 : 2
    for j in start:n_particles
        @inbounds simulate_initial!(X[j,1], model, data, θ)
    end

    for j in 1:n_particles
        @inbounds W[j,1] = log_observation_density(X[j,1], model, 1, data, θ)
    end

    @views logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    wnorm .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t - 1)
        # resample!(a, W[:, t - 1], resampling, conditional)
        resample!(a, wnorm, resampling, !(conditional == :no))
        if conditional == :as
            ancestor_weights!(wnorm, X[1, t], X[:, t-1], model, t-1, data, θ)
            a[1] = sample_one_index(wnorm)
        end
        # if ancestorsampling #&& conditional
        #     wancestor = compute_ancestor_weights(X[:,t-1], X[1,t], wnorm, wancestor, model, t, data, θ)  # (xⁱₜ₋₁,xₜʳ,wⁱₜ₋₁,...) Returns normalized w
        #     a[1] = sample_one_index(wancestor)
        # end
        
        for j in start:n_particles
            simulate_transition!(X[j,t], X[a[j],t - 1], model, t - 1, data, θ)
        end

        for j in 1:n_particles
            @inbounds W[j,t] = log_observation_density(X[j,t], model, t, data, θ)
        end
        @views logΣexp = logsumexp(W[:, t])
        ll += logΣexp - log(n_particles)
        @views wnorm .= exp.(W[:, t] .- logΣexp)
    end

    if !(conditional == :no)
        idx = sample_one_index(wnorm)
        condition_on_particle!(storage, idx)
    end
    
    # if conditional
    #     println("Reference trajectory before generating new is ")
    #     println(X[1,:])
    #     println("state trajectory matrix is ")
    #     println(X)
    #     finalInd = sample_one_index(wnorm); # Previously W[:, T]
    #     Xref = generate_trajectory(A, X, ref, finalInd);  # Add Xref here if want to store Xref separately between runs
    #     return Xref;
    # end
    
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
    resampling::Resampling=MultinomialResampling(), 
    conditional::Symbol=:no,      
    # contitional::Bool=false
    n_particles::Int=particle_count(storage) )

    @assert n_particles <= particle_count(storage)
    T = length(data.y)
    X = view(storage.X, 1:n_particles, :);
    W = view(storage.W, 1:n_particles, :);
    A = view(storage.A, 1:n_particles, :);
    Ψ = view(storage.Ψ, 1:n_particles, :);
    wnorm = view(storage.wnorm, 1:n_particles);

    ll = 0

    start = conditional == :no ? 1 : 2 # Skip reference (index 1) if conditional

    # Simulate initial state
    for j in start:n_particles
        # @inbounds simulate_initial!(X[j,1], model, data, θ)
        @inbounds simulate_proposal!(X[j, 1], model, data, θ)
    end

    # Compute weights
    for j in 1:n_particles
        @inbounds Ψ[j, 1] = log_potential(X[j, 1], model, 1, data, θ)
        @inbounds W[j, 1] = (
            log_observation_density(X[j, 1], model, 1, data, θ)
            + log_initial_density(X[j, 1], model, data, θ)
            + Ψ[j, 1]
            - log_proposal_density(X[j, 1], model, data, θ))
    end

    logΣexp = logsumexp(W[:, 1])
    ll += logΣexp - log(n_particles)
    @views wnorm .= exp.(W[:, 1] .- logΣexp)

    for t in 2:T
        a = view(A, :, t - 1)
        resample!(a, wnorm, resampling, !(conditional == :no))
        if conditional == :as
            ψ = view(Ψ, :, t - 1)
            ancestor_weights!(wnorm, X[1, t], X[:, t-1], model, t-1, data, θ, ψ)
            a[1] = sample_one_index(wnorm)
        end

        for j in start:n_particles
            # @inbounds simulate_transition!(X[j, t], X[a[j], t - 1], model, t - 1, data, θ)
            @inbounds simulate_proposal!(X[j, t], X[a[j], t-1], model, t, data, θ)
        end

        for j in 1:n_particles
            @inbounds Ψ[j, t] = log_potential(X[j, t], model, t, data, θ)
            @inbounds W[j, t] = (
                   log_transition_density(X[j, t], X[a[j], t-1], model, t-1, data, θ)
                 + log_observation_density(X[j, t], model, t, data, θ)
                 + Ψ[j, t]
                 - Ψ[a[j], t-1]
                 - log_proposal_density(X[j, t], X[a[j], t-1], model, t, data, θ))
        end
        @views logΣexp = logsumexp(W[:, t])
        ll += logΣexp - log(n_particles)
        @views wnorm .= exp.(W[:, t] .- logΣexp)
    end

    if !(conditional == :no)
        idx = sample_one_index(wnorm)
        condition_on_particle!(storage, idx)
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
