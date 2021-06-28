"""
General state-space model
Expects the following function
* `observation_function(x, e, model, t, data, θ)`
* `transtiion_function(x, v, model, t, data, θ)`
"""
abstract type AbstractSSM{P<:FloatParticle} <: AbstractModel{P} end

"""
Observation function that an AbstractSSM must implent
```math
yₜ = gₜ(xₜ, eₜ)
```
Arguments:
* `xₜ::AbstractVector`: current state
* `e::AbstractVector`: observation noise
* `model::AbstractSSM`: ssm model
* `t`: time index
* `data`: data needed to run filter, eg. input u and output y
* `θ`: parameter for the model
"""
function observation_function(xₜ, e, ::AbstractSSM, t, data, θ) end
function observation_function(xₜ, model::AbstractSSM, t, data, θ)
    return observation_function(xₜ, mean(observation_noise(xₜ, model, t, data, θ)), model, t, data, θ)
end

"""
Transition function that AbstractSSM must implement
xₜ₊₁ = fₜ(xₜ, vₜ)
Arguments:
* `xₜ::AbstractVector`: current state
* `w::AbstractVector`: process noise
* `model::AbstractSSM`: ssm model
* `t`: time index
* `data`: data needed to run filter, eg. input u and output y
* `θ`: parameter for the model
"""
function transition_function(xₜ, w, model::AbstractSSM, t, data, θ) end
function transition_function(xₜ, model::AbstractSSM, t, data, θ)
    return transition_function(xₜ, mean(transition_noise(xₜ, model, t, data, θ)), model, t, data, θ)
end

"""
Returns the initial probability density.
The returned value must the following methods:
 * rand!(pdf)
 * logdf()
And for Guassian fitlering:
 * mean(pdf)
 * cov(pdf)
"""
function initial_density(model::AbstractSSM, data, θ) end
function observation_noise(xₜ, model::AbstractSSM, t, data, θ) end
function transition_noise(xₜ, model::AbstractSSM, t, data, θ) end

"""
Nedded for moment mathcing gaussian filters, eg. Kalman fitler.
"""
initial_mean(model, data, θ)= mean(initial_density(model, data, θ))
initial_covariance(model, data, θ) = cov(initial_density(model, data, θ))
transition_noise_covariance(xₜ, model, t, data, θ) = cov(transition_noise(xₜ, model, t, data, θ))
observation_noise_covariance(xₜ, model, t, data, θ) = cov(observation_noise(xₜ, model, t, data, θ))
# simulate_transition_noise(xₜ, model, t, data, θ) = rand(transition_noise(xₜ, model, t, data, θ))
# simulate_observation_noise(xₜ, model, t, data, θ) = rand(observation_noise(xₜ, model, t, data, θ))

# Jacobians
function transition_state_jacobian(xₜ, model::AbstractSSM, t, data, θ)
    dfdx = ForwardDiff.jacobian(
        μ -> transition_function(μ, model, t, data, θ),
        SVector(xₜ)
    )
    return dfdx
end

function transition_noise_jacobian(xₜ, model::AbstractSSM, t, data, θ)
    return ForwardDiff.jacobian(
        μ -> transition_function(SVector(xₜ), μ, model, t, data, θ),
        mean(transition_noise(xₜ,  model, t, data, θ))
    )
end

function observation_state_jacobian(xₜ, model::AbstractSSM, t, data, θ)
    dhdx = ForwardDiff.jacobian(
            μ->observation_function(μ, model, t, data, θ),
            SVector(xₜ)
        )
    return dhdx
end

function observation_noise_jacobian(xₜ, model::AbstractSSM, t, data, θ)
    ForwardDiff.jacobian(
        μ->observation_function(SVector(xₜ), μ, model, t, data, θ),
        mean(observation_noise(xₜ, model, t, data, θ))
    )
end

########

@inline function simulate_initial!(p, model::AbstractSSM, data, θ)
    d = initial_density(model, data, θ)
    return rand!(d, p.x)
end

function simulate_transition!(xₜ₊₁, xₜ, model::AbstractSSM, t, data, θ)
    d =  transition_noise(xₜ, model, t, data, θ)
    xₜ₊₁.x .= transition_function(SVector(xₜ.x), rand(d), model, t, data, θ)
    return xₜ₊₁
end

function simulate_observation(xₜ, model::AbstractSSM, t, data, θ)
    d = observation_noise(xₜ, model, t, data, θ)
    return observation_function(xₜ.x, rand(d), model, t, data, θ)
end


function log_initial_density(xₜ, model::AbstractSSM, data, θ)
    d = initial_density(model, data, θ)
    return logpdf(d ,xₜ.x)
end

""""
Transition density for additive noise.
xₜ₊₁ = fₜ(xₜ) + vₜ
"""
function log_transition_density(xₜ₊₁, xₜ, model::AbstractSSM, t, data, θ)
    d = transition_noise(xₜ, model, t, data, θ)
    v = SVector(xₜ₊₁.x) - transition_function(SVector(xₜ.x), model, t, data, θ)
    return logpdf(d, v)
end

"""
Observation density for additive observation noise
yₜ = hₜ(xₜ) + eₜ
"""
function log_observation_density(xₜ, model::AbstractSSM, t, data, θ)
    d = observation_noise(xₜ, model, t, data, θ)
    ŷ = observation_function(SVector(xₜ.x), model, t, data, θ)
    return logpdf(d, data.y[t]- ŷ)
end
