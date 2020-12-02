"""
An abstract type for a general statespace model
that does not (have to) admit a density for the 
transition or observation distributions.

An SSM expects different functions for different filters.

To __simulate__ a model, the following are needed:
* `simulate_initial!(x <: Particle, model <: SSM, data, parameter)`:
Sample from the initial distribution p(x₁).
* `simulate_transition!(xnext <: Particle, xcurr <: Particle, model <: SSM, t::Integer, data, parameter)`:
Simulate from the transition distribution pₜ(xₜ₊₁ | xₜ).
* `simulate_observation!(y::AbstractVector{<:AbstractFloat}, x<:Particle, model <: SSM, t::Integer, data, parameter)`:
Simlulate from the data distribution pₜ(yₜ | xₜ).

To run a __bootstrap particle filter__, you also need:
`log_observation_density(y::AbstractVector{<:AbstractFloat}, x<:Particle, t::Integer, data, parameter)`:
Evaluate the log density for p(yₜ | xₜ).

To run a __general particle filter__, you also need
* `log_initial_density(x <: Particle, model <: SSM, data, parameter)`:
Evaluate the log density for the initial distributrion p(x₁).
* `log_transition_density(xnext <: Particle, , xcurr <: Particle, model <: SSM, t::Integer, data, parameter)`,
evaluating the log density for the transition ditribution pₜ(xₜ₊₁ | xₜ).
"""
abstract type SSM{P<:Particle} end

"""
Return the particle type associated with a GenericSSM.
"""
function particletype(::SSM{P}) where P <: Particle
    P;
end

"""
Simulate `length(y)` amount of observations from a generic state space model
(SSM).
Arguments:
* `y`: A preallocated vector of vectors where simulated observations should be
stored.
* `model`: A SSM object.
* `data`: Data needed by the model`.
* `θ`: Parameters of the model.
"""
function simulate!(y::AVec{<:AVec{<:AFloat}}, model::SSM{P}, data, θ) where {P}
    @assert !isempty(y) "the vector `y` must not be empty.";
    N = length(y);
    xcur = P(); xnext = P();

    simulate_initial!(xcur, model, data, θ);
    simulate_observation!(y[1], xcur, model, 1, data, θ);
    for i in 2:N
        simulate_transition!(xnext, xcur, model, i-1, data, θ);
        simulate_observation!(y[i], xnext, model, i, data, θ);
        copy!(xcur, xnext);
    end
    nothing;
end


"""
Simulate both the observations and the latent state.
"""
function simulate!(y::AVec{<:AVec{<:AFloat}},
                   x::AVec{P},
                   model::SSM{P},
                   data, θ) where {P}
    @assert !isempty(y) "the vector `y` must not be empty.";
    @assert !isempty(x) "the vector `x` must not be empty.";
    N = length(y);
    @assert length(x) == N "the lengths of `x` and `y` do not match.";

    simulate_initial!(x[1], model, data, θ);
    simulate_observation!(y[1], x[1], model, 1, data, θ);
    for i in 2:N
        simulate_transition!(x[i], x[i - 1], model, i, data, θ);
        simulate_observation!(y[i], x[i], model, i, data, θ);
    end
    nothing;
end

function transition_state_jacobian(x, model::SSM, t, data, θ)
    ForwardDiff.jacobian(μ->transition_function(μ, model, t, data, θ), x)
end

function observation_state_jacobian(x, model::SSM, t, data, θ)
    ForwardDiff.jacobian(μ->observation_function(μ, model, t, data, θ), x)
end

function initial_mean(::SSM, data, θ)
    @unpack μ0 = θ
    μ0
end

function initial_covariance(::SSM, data, θ)
    @unpack Σ0 = θ
    Σ0
end

function transition_covariance(x, ::SSM, t, data, θ)
    @unpack Q = θ
    Q
end

function observation_covariance(x, ::SSM, t, data, θ)
    @unpack R = θ
    R
end