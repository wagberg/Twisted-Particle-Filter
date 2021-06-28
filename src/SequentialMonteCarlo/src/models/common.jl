"""
An abstract type for a general model
that does not (have to) admit a density for the
transition or observation distributions.

A model expects different functions for different filters.

To run a bootstrap particle filter the following methods are needed:
* `simulate_initial!(x₁<:Particle, model<:AbstractModel, data, θ)`:
Sample from the initial distribution p(x₁).
* `simulate_transition!(xₜ₊₁<:Particle, xₜ<:Particle, model<:AbstractModel, t, data, θ)`:
Simulate from the transition distribution pₜ(xₜ₊₁ | pₜ).
* `log_observation_density(pₜ<:Particle, model<:AbstractModel, t, data, θ)`:
Evaluate the log density for p(yₜ | xₜ).

To __simulate__ a model, the following are needed:
* `simulate_observation(xₜ<:Particle, model<:AbstractModel, t, data, θ)`:
Simlulate from the data distribution pₜ(yₜ | xₜ).

To run a __general particle filter__, you also need
* `log_initial_density(x₁<:Particle, model<:AbstractModel, data, θ)`:
Evaluate the log density for the initial distributrion p(x₁).
* `log_transition_density(xₜ₊₁<:Particle, xₜ<:Particle, model<:AbstractModel, t, data, θ)`,
Evaluate the log density for the transition ditribution pₜ(xₜ₊₁ | xₜ).
"""
abstract type AbstractModel{P<:Particle} end

"""
Return the particle type associated with an AbstractSSM.
"""
particletype(::AbstractModel{P}) where P <: Particle = P

"""
Returns the names of the states
"""
statenames(::AbstractModel{P}) where P <: Particle = statenames(P)

"""
Simulate the initial density in-place
Arguments:
- `x₁`: particle
- `model`: state space model
- `data`: data needed for evaluation
- `θ`: parameter
"""
function simulate_initial!(x₁, model::AbstractModel, data, θ) end

"""
Simulate next state
Arguments:
* `xₜ₊₁`: next particle
* `xₜ`: current particle
* `model`: model to simulate
* `t`: time step
* `data`: data
* `θ`: parameter
"""
function simulate_transition!(xₜ₊₁, xₜ, model::AbstractModel, t, data, θ) end

"""
Logarithm of the observation probability density function
Arguments:
* `pₜ`: particle
* `model`: state space model
* `t`: time step
* `data`: data
* `θ`: parameter
"""
function log_observation_density(xₜ, model::AbstractModel, t, data, θ) end


"""
Simulate observation from state space model
Arguments:
* `pₜ`: particle
* `model`: state space model
* `t`: time step
* `data`: data
* `θ`: parameter
"""
function simulate_observation(pₜ, model::AbstractModel, t, data, θ) end

"""
Convert a model parameter to a vector
"""
function par_to_vec(θ) end

"""
Convert a vector to a model parameter
"""
function vec_to_par(model::AbstractModel, v) end

"""
Simulate observations from a state space model (SSM).
Arguments:
* `model`: A state space model.
* `data`: Data needed by the model`.
* `θ`: Parameters of the model.
* `T`: Number of observations
"""
function simulate(model, data, θ, T::Int = 100)
    P = particletype(model)
    p = [P() for t in 1:T]
    simulate_initial!(p[1], model, data, θ)
    y = [simulate_observation(p[1], model, 1, data, θ)]
    for t in 2:T
        simulate_transition!(p[t], p[t-1], model, t-1, data, θ)
        push!(y, simulate_observation(p[t], model, t, data, θ))
    end
    return y, p
end

"""
Deep copy of particles from src to dest.
"""
function copy_trajectory!(dest::AbstractArray{P}, src::AbstractArray{P}) where P <: Particle
    @assert length(dest) == length(src)
    for i in eachindex(dest)
        copy!(dest[i], src[i])
    end
end
