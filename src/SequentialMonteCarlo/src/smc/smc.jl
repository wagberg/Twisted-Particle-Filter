abstract type AbstractSMC end

"""
Returns the normalized
"""
normalized_weights(f::AbstractSMC) = normalized_weights(_storage(f))
normalize!(f::AbstractSMC, t) = normalize!(_storage(f), t)
_n_particles(f::AbstractSMC) = _n_particles(_storage(f))
_n_particles!(f::AbstractSMC, t) = _n_particles!(_storage(f), t)
_model(f::AbstractSMC) = _model(_storage(f))
_data(f::AbstractSMC) = _data(_storage(f))
states(f::AbstractSMC) = states(_storage(f))
states(f::AbstractSMC, t) = states(_storage(f), t)
weights(f::AbstractSMC) = weights(_storage(f))
weights(f::AbstractSMC, t) = weights(_storage(f), t)
potentials(f::AbstractSMC) = potentials(_storage(f))
potentials(f::AbstractSMC, t) = potentials(_storage(f), t)
ancestors(f::AbstractSMC) = ancestors(_storage(f))
ancestors(f::AbstractSMC, t) = ancestors(_storage(f), t)

"""
Initialize particle storage. Typically sets the initial weights to 1/N.
"""
function initialize!(f::AbstractSMC, θ)
    fill!(weights(f, 1)[1], -log(_n_particles(f)))
end

"""
Function to run before end, eg. to sample reference trajectory in conditional particle filter
(or run backwards sweep in particle smoother?)
"""
function finalize!(f::AbstractSMC, θ) end

"""
Sample initial particles x₁ ∼ q₁.
"""
function simulate_initial!(::AbstractSMC, θ) end

"""
Compute initial weights w₁ = γ₁/q₁
"""
function weight!(::AbstractSMC, θ) end
"""
Compute weights wₜ = wₜ₋₁ γₜ(xₜ)λₜ₋₁(xₜ,xₜ₋₁) / γₜ₋₁(xₜ₋₁)κₜ(xₜ₋₁,xₜ)
"""
function weight!(::AbstractSMC, t, θ) end

"""
Propagate particle from iteration `t-1` to `t`, ie. sample kernel K(xₜ₋₁, dxₜ).
"""
function propagate!(::AbstractSMC, t, θ) end

"""
Sample ancestors at time `t-1`. Store ancestor indices in `ancestors(f, t-1)`. Indices are
sampled with weights in `normalized_weights(f)` which are assumed to sum to 1. Resampled
log weights are stored in `weights(f, t)`.
"""
function resample!(f::AbstractSMC, t, θ)
    logWₜ, logWₜ₋₁ = weights(f, t)
    resample!(
        _resampler(f),
        ancestors(f,t-1),
        normalized_weights(f),
        logWₜ₋₁,
        logWₜ)
end


function run_filter!(f::AbstractSMC, θ)
    @timeit_debug "initialize!" initialize!(f, θ) # set initial weights to 1/N
    @timeit_debug "simulate_initial!" simulate_initial!(f, θ) # fill s.X with samples from proposal
    @timeit_debug "initial_weight!" weight!(f, θ) # compute weights in s.W[:,1]
    normalize!(f, 1) # normalize s.W[:,1] store exp weights in ps.w

    T = length(_data(f).y) #length of data
    for k in 2:T
        # compute ancestors at time t-1 to propagate to time t using weights in p.w
        # storing in p.A[:,t-1]. If conditional, keep first particle be only sampling
        # ps.A[2:end,t-1] and setting ps.A[1,t-1] = 1. Also copy weights from p.W[:,t-1] to
        # p.W[:,t] or set p.W[:,t] = -log(n_particles)
        resample!(f, k, θ)
        # Propagate particles from p.X[a[i],t-1] -> p.X[i,t] by sampling proposal.
        @timeit_debug "propagate!" propagate!(f, k, θ)
        # Compute weights = wₜ₋₁*γₜ/ γₜ₋₁  store in p.W[:,t]
        @timeit_debug "weight!" weight!(f, k, θ)
        normalize!(f, k) # normalize s.W[:,t] and store exp weights in p.w
    end
    finalize!(f, θ)
end

include("potentials/potentials.jl")
include("proposals/proposals.jl")
include("storage.jl")
include("resamplers/resamplers.jl")
include("particle_filters/particle_filters.jl")



# function run_filter!(s::ParticleStorage, f::AbstractParticleFilter, θ)
#     @assert n_particles <= capacitys(s)
#     _n_particles!(s, n_particles)
#     data = _data(s)
#     ll = _ll(s)
#     T = length(data.y)

#     # Reset time index and starting weights
#     reset!(s)
#     # Accumulator for log likelihood
#     llc = 0.0
#     # Simulate initial state
#     propagate_particles!(pf, θ, conditional)
#     # Compute unnormalized weights
#     compute_weights!(pf, θ)
#     # Normalize weights and estimate log likelihood p(y₁)
#     ll[1] = llc += normalize!(pf)
#     # Sample ancstors
#     if should_resample(pf, ess_threshold)
#         # Sample ancestors for the current time
#         resample!(pf)
#     else
#         # Set ancestors
#         ancestors(pf) .= 1:get_n_particles(pf)
#         # Propagate weights
#         weights(pf, 2) .= weights(pf, 1)
#     end

#     for _ in 2:T
#         # Increase time index in filter
#         increase_t!(pf)
#         t = get_t(pf)
#         if conditional == :as # Maybe only run when resampling?
#             # sample ancestors for first particle
#             X = states(pf)
#             Xp = states(pf, t-1)
#             ancestor_weights!(normalized_weights(pf), X[1], Xp, model, t-1, data, θ)
#             a[1] = sample_one_index(normalized_weights(pf))
#         end
#         @timeit_debug "propagate_particles!" propagate_particles!(pf, θ, conditional)
#         @timeit_debug "compute_weights!" compute_weights!(pf, θ)
#         ll[t] = llc += normalize!(pf)

#         if should_resample(pf, ess_threshold)
#             resample!(pf)
#         else
#             ancestors(pf) .= 1:n_particles
#             # Propagate weights
#             weights(pf, t+1) .= weights(pf, t)
#         end
#     end

#     if !(conditional == :no)
#         idx = sample_one_index(wnorm)
#         condition_on_particle!(storage, idx)
#     end

# end

# function run_smoother!(s::ParticleStorage, f::AbstractParticleSmoother, θ) end



# @inline should_resample(pf::AbstractParticleFilter, thres) = (ess(normalized_weights(pf)) <= get_n_particles(pf)*thres)

# (pf::AbstractParticleFilter)(args...; kwargs...) = run_filter!(pf, args...; kwargs...)

# LinearAlgebra.normalize!(pf::AbstractParticleFilter)= logsumexp!(weights(pf), normalized_weights(pf))
# """
# Return number of preallocated particles
# """
# particle_count(pf::AbstractParticleFilter) = size(all_states(pf), 1)

# """
# Returns the type of the potential
# """
# potential_type(pf::AbstractParticleFilter) = potential_type(particle_filter(pf))
# """
# Returns the type of the proposal
# """
# proposal_type(pf::AbstractParticleFilter) = proposal_type(particle_filter(pf))



# """
# Compute ancestor weights bpf
# """
# function ancestor_weights!(w, pcurr, pprev, model, t, data, θ)
#     for i in eachindex(pprev)
#         w[i] = log(w[i]) + log_transition_density(pcurr, pprev[i], model, t, data, θ)
#     end
#     logΣexp = logsumexp(wnorm)
#     wnorm .= exp.(wnorm .- logΣexp)
#     nothing
# end

# """
# Propagate particles from time t-1 to t
# """
# function propagate_particles!(pf, θ, conditional)
#     t = get_t(pf)
#     proposal = get_proposal(pf)
#     data = get_data(pf)
#     if t == 1 # Inital state
#         X = states(pf)
#         conditional !== :no || rand!(proposal, X[1], data, θ)
#         @inbounds for i in 2:length(X)
#             rand!(proposal, X[i], data, θ)
#         end
#     else
#         X = states(pf)
#         Xp = states(pf, t-1)
#         a = ancestors(pf, t-1)
#         conditional !== :no || rand!(proposal, X[1], Xp[a[1]], t-1, data, θ)
#         @timeit_debug "rand proposal" @inbounds for i in 2:length(X)
#             rand!(proposal, X[i], Xp[a[i]], t-1, data, θ)
#         end
#     end
#     return pf
# end

# """
# Compute weights at time t
# """
# function compute_weights!(pf, θ)
#     data = get_data(pf)
#     model = get_model(pf)
#     potential = get_potential(pf)
#     proposal = get_proposal(pf)

#     t = get_t(pf)
#     W = weights(pf)
#     Ψ = potentials(pf)
#     X = states(pf)
#     if t == 1 # Initial time
#         @inbounds for i in eachindex(W)
#             Ψ[i] = log_potential(potential, X[i], t, data, θ)
#             log_obs_dens = log_observation_density(X[i], model, t, data, θ)
#             log_init_dens = log_initial_density(X[i], model, data, θ)
#             log_proposal = logpdf(proposal, X[i], data, θ)
#             W[i] += log_obs_dens + log_init_dens + Ψ[i] - log_proposal
#         end
#     else
#         a = ancestors(pf, t-1)
#         Xp = states(pf, t-1) # previous states
#         Ψp = potentials(pf, t-1) # previous potentials
#         @inbounds for i in eachindex(W)
#             @timeit_debug "log potential" Ψ[i] = log_potential(potential, X[i], t, data, θ)
#             @timeit_debug "transition_density" log_trans_dens = log_transition_density(X[i], Xp[a[i]], model, t-1, data, θ)
#             @timeit_debug "observation_ensity" log_obs_dens = log_observation_density(X[i], model, t, data, θ)
#             @timeit_debug "log proposal" log_proposal = logpdf(proposal, X[i], Xp[a[i]], t-1, data, θ)
#             W[i] += log_trans_dens + log_obs_dens + Ψ[i] - Ψp[a[i]] - log_proposal
#         end
#     end
#     return pf
# end

# function reset_weights!(pf::AbstractParticleFilter)
#     n_particles = get_n_particles(pf)
#     fill!(weights(pf), -log(n_particles))
#     return pf
# end

# function reset_all_weights!(pf::AbstractParticleFilter)
#     n_particles = get_n_particles(pf)
#     fill!(all_weights(pf), -log(n_particles))
#     return pf
# end

# function resample!(pf::AbstractParticleFilter)
#     resampler = get_resampler(pf)
#     w = normalized_weights(pf)
#     a = ancestors(pf)
#     resample!(resampler, a, w)
#     return pf
# end

# function reset!(pf::AbstractParticleFilter)
#     reset_t!(pf)
#     reset_all_weights!(pf)
#     return pf
# end

# function run_filter!(pf::AbstractParticleFilter, θ, n_particles=get_n_particles(pf), ess_threshold=0.5, conditional=:no)
#     @assert n_particles <= particle_count(pf)
#     set_n_particles!(pf, n_particles)
#     data = get_data(pf)
#     ll = get_ll(pf)
#     T = length(data.y)

#     # Reset time index and starting weights
#     reset!(pf)
#     # Accumulator for log likelihood
#     llc = 0.0
#     # Simulate initial state
#     propagate_particles!(pf, θ, conditional)
#     # Compute unnormalized weights
#     compute_weights!(pf, θ)
#     # Normalize weights and estimate log likelihood p(y₁)
#     ll[1] = llc += normalize!(pf)
#     # Sample ancstors
#     if should_resample(pf, ess_threshold)
#         # Sample ancestors for the current time
#         resample!(pf)
#     else
#         # Set ancestors
#         ancestors(pf) .= 1:get_n_particles(pf)
#         # Propagate weights
#         weights(pf, 2) .= weights(pf, 1)
#     end

#     for _ in 2:T
#         # Increase time index in filter
#         increase_t!(pf)
#         t = get_t(pf)
#         if conditional == :as # Maybe only run when resampling?
#             # sample ancestors for first particle
#             X = states(pf)
#             Xp = states(pf, t-1)
#             ancestor_weights!(normalized_weights(pf), X[1], Xp, model, t-1, data, θ)
#             a[1] = sample_one_index(normalized_weights(pf))
#         end
#         @timeit_debug "propagate_particles!" propagate_particles!(pf, θ, conditional)
#         @timeit_debug "compute_weights!" compute_weights!(pf, θ)
#         ll[t] = llc += normalize!(pf)

#         if should_resample(pf, ess_threshold)
#             resample!(pf)
#         else
#             ancestors(pf) .= 1:n_particles
#             # Propagate weights
#             weights(pf, t+1) .= weights(pf, t)
#         end
#     end

#     if !(conditional == :no)
#         idx = sample_one_index(wnorm)
#         condition_on_particle!(storage, idx)
#     end
#     return nothing
# end




# include("full_particle_filter.jl")
# include("bootstrap_particle_filter.jl")
# include("twisted_particle_filter.jl")

# function reduce_trajectory(ps::ParticleStorage, f::Function)
#     w = similar(ps.W, size(ps.W, 1))
#     x = Vector{typeof(toSVector(ps.X[1]))}()
#     for (t, (X,W)) in enumerate(zip(eachcol(ps.X), eachcol(ps.W)))
#         SequentialMonteCarlo.expnormalize!(w, W)
#         push!(x, sum(f.(toSVector.(X), t).*w))
#     end
#     return x
# end
# mean_trajectory(ps::ParticleStorage) = reduce_trajectory(ps, (x, t)->x)
# function mean_and_std(ps::ParticleStorage)
#     m = mean_trajectory(ps)
#     s = map(x->sqrt.(x), reduce_trajectory(ps, (x,t)->(x.-m[t]).^2))
#     return m, s
# end

# function sample_one_index(w::AVec{<:AbstractFloat}, rng::AbstractRNG = Random.GLOBAL_RNG)
#     # TODO: Add check that w is normalized
#     u = rand(rng)
#     csum = zero(Float64)
#     N = length(w)
#     for i in 1:N
#         csum += @inbounds w[i]
#         if u <= csum
#             return i
#         end
#     end
#     N   # return N if loop terminates
# end

# function generate_trajectory(A::AMat{Int}, X::AMat{<:Particle}, ref::AVec{Int}, finalInd::Int) # Add Xref::AVec{<:Particle}, as argumet if want to store Xref separately
#     # Change X[1,t] to Xref[t] if want to store reference separately
#     T = size(X, 2)

#     ref[T] = finalInd  # Save sampled trajectory on last place in ref
#     X[1,T].x .= X[finalInd,T].x
#     for t in (T-1):-1:1
#         ref[t] = A[ref[t+1],t]
#         X[1,t].x .= X[ref[t],t].x
#     end
#     #X[1,:]
# end

# #function generate_trajectory(A::AMat{Int}, X::AMat{<:Particle}, ref::AVec{Int}, finalInd::Integer) # Add Xref::AVec{<:Particle}, as argumet if want to store Xref separately
# #    # Change X[1,t] to Xref[t] if want to store reference separately
# #    println(typeof(X))
# #    T = size(X, 2)
# #
# #    ref[T] = finalInd  # Save sampled trajectory on last place in ref
# #    X[1,T] = X[finalInd,T]
# #    for t in (T-1):-1:1
# #        ref[t] = A[ref[t+1],t]
# #        X[1,t] = X[ref[t],t]
# #    end
# #    #X[1,:]
# #end

# function compute_ancestor_weights(Xi::AVec{<:Particle}, Xref::Particle, W::AVec{Float64}, wancestor::AVec{Float64}, model::AbstractSSM, t::Integer, data, θ) # (anc, xⁱₜ₋₁, xₜʳ,wⁱₜ₋₁ )
#     n_particles = length(W)
#     for i in 1:n_particles
#         wancestor[i] = log(W[i]) + log_transition_density(Xref, Xi[i], model, t, data, θ)  # For SSM, no twisting
#     end
#     logΣexp = logsumexp(wancestor)
#     wancestor .= exp.(wancestor .- logΣexp)
#     #return wancestor
# end

# """
# Compute ancestor weights bpf
# """
# function ancestor_weights!(wnorm, pcurr, pprev, model, t, data, θ)
#     for i in eachindex(pprev)
#         wnorm[i] = log(wnorm[i]) + log_transition_density(pcurr, pprev[i], model, t, data, θ)
#     end
#     logΣexp = logsumexp(wnorm)
#     wnorm .= exp.(wnorm .- logΣexp)
#     nothing
# end

# """
# Compute ancestor weights tpf
# """
# function ancestor_weights!(wnorm, pcurr, pprev, model, t, data, θ, Ψ::AVec{Float64})
#     for i in eachindex(pprev)
#         wnorm[i] = log(wnorm[i]) + log_transition_density(pcurr, pprev[i], model, t, data, θ) - Ψ[i]
#     end
#     logΣexp = logsumexp(wnorm)
#     wnorm .= exp.(wnorm .- logΣexp)
#     nothing
# end

# """
# Condition on particle ending in X[idx, T] by moving the whole
# trajectory to X[1, :].
# """
# function condition_on_particle!(ps::ParticleStorage, idx::Integer)
#     T = size(ps.X, 2)
#     swap!(ps.X[1, T], ps.X[idx, T])

#     for t = T-1:-1:1
#         idx = ps.A[idx, t]
#         swap!(ps.X[1, t], ps.X[idx, t])
#     end
# end



# function DataFrames.DataFrame(ps::ParticleStorage, model)
#     m, s = transpose.(reinterpret.(reshape, Float64, SequentialMonteCarlo.mean_and_std(ps)))
#     sn = statenames(model)
#     return df = DataFrame(hcat(m,s), vec(["E" "σ"] .* string.(sn)))
# end
