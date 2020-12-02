const StochasticVolatilityParticle = FloatParticle{1}
"""
Stochastic volatility model:

xₜ | xₜ₋₁ ~ N(xₜ ; ϕxₜ₋₁, σ²)
yₜ | xₜ ~N(yₜ ; 0, β²exp(xₜ))
x₁ ~ N(x₁ ; μ₁, σ₁²)

Parameters:
* `μ₁` - Initial state mean
* `logσ₁` - Initial state std log(σ₁)
* `logϕ` - log(ϕ)
* `logσ` - log(σ)
* `logβ` - log(β)
"""
struct StochasticVolatility <: SSM{StochasticVolatilityParticle}
end

@with_kw struct SVParameter
    μ₁::Float64    = 0.0
    logσ₁::Float64 = log(1.00)
    logϕ::Float64  = log(0.98)
    logσ::Float64  = log(0.16)
    logβ::Float64  = log(0.70)
end


function SequentialMonteCarlo.transition_function(x, ::StochasticVolatility, t, data, θ)
    @unpack logϕ = θ
    exp(logϕ).*x
end

function SequentialMonteCarlo.observation_function(x, ::StochasticVolatility, t, data, θ)
    return zero(x)
end


function SequentialMonteCarlo.initial_mean(::StochasticVolatility, data, θ)
    @unpack μ₁ = θ
    [μ₁]
end

function SequentialMonteCarlo.initial_covariance(::StochasticVolatility, data, θ)
    @unpack logσ₁ = θ
    hcat(exp(logσ₁))
end

function SequentialMonteCarlo.transition_covariance(x, ::StochasticVolatility, t, data, θ)
    @unpack logσ = θ
    hcat(exp(logσ))
end

function SequentialMonteCarlo.observation_covariance(x, ::StochasticVolatility, t, data, θ)
    @unpack logβ = θ
    @inbounds hcat(exp(2*logβ + x[1]))
end
