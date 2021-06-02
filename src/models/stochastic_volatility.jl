using Parameters
"""
Stochastic volatility model:

xₜ | xₜ₋₁ ~ N(xₜ ; ϕxₜ₋₁, σ²)
yₜ | xₜ ~N(yₜ ; 0, β²exp(xₜ))
x₁ ~ N(x₁ ; μ₁, σ₁²)

Functional model:
xₜ₊₁ = f(xₜ, vₜ) = ϕxₜ + σvₜ        vₜ ∼ N(0, 1)
  yₜ = h(xₜ, eₜ) = β*exp(xₜ/2)eₜ    eₜ ∼ N(0, 1)
  x₁ ∼ N(μ₁, σ₁²)

Parameters:
* `μ₁` - Initial state mean
* `logσ₁` - Initial state std log(σ₁)
* `logϕ` - log(ϕ)
* `logσ` - log(σ)
* `logβ` - log(β)
"""
const StochasticVolatilityParticle{T} = FloatParticle{1, T}

struct StochasticVolatility{T} <: AbstractSSM{StochasticVolatilityParticle{T}}
end

StochasticVolatility() = StochasticVolatility{Float64}()

@with_kw struct SVParameter{T}
    μ₁::T    = 0.0
    logσ₁::T = log(1.00)
    logϕ::T  = log(0.98)
    logσ::T  = log(0.16)
    logβ::T  = log(0.70)
end
SVParameter(;kwargs...) = SVParameter{Float64}(;kwargs...)

SequentialMonteCarlo.observation_noise_dimension(::SVParameter) = 1
SequentialMonteCarlo.transition_noise_dimension(::SVParameter) = 1

function SequentialMonteCarlo.initial_mean(::StochasticVolatility{T}, data, θ) where {T}
    @unpack μ₁ = θ
    return SVector{1, T}(μ₁)
end

function SequentialMonteCarlo.initial_covariance(::StochasticVolatility{T}, data, θ) where {T}
    @unpack logσ₁ = θ
    return SVector{1, T}(exp(logσ₁))
end

function SequentialMonteCarlo.simulate_initial(model::StochasticVolatility{T}, data, θ) where {T}
    @unpack μ₁, logσ₁ = θ
    return μ₁ .+ exp(logσ₁)*randn(SVector{1, T})
end

function SequentialMonteCarlo.transition_function(x, v, ::StochasticVolatility, t, data, θ)
    @unpack logϕ, logσ = θ
    return exp(logϕ).*x .+ logσ.*v
end

function SequentialMonteCarlo.transition_noise_covariance(x, ::StochasticVolatility{T}, t, data, θ) where {T}
    return LowerTriangular(one(SMatrix{1,1,T}))
end

function SequentialMonteCarlo.simulate_transition_noise(x, model::StochasticVolatility{T}, t, data, θ) where {T}
    return randn(SVector{1, T})
end


function SequentialMonteCarlo.observation_function(x, e, ::StochasticVolatility{T}, t, data, θ) where {T}
    @unpack logβ = θ
    return zero(SVector{1, T}) .+ exp.(logβ .+ 0.5*x).*e
end

function SequentialMonteCarlo.observation_noise_covariance(x, ::StochasticVolatility{T}, t, data, θ) where {T}
    return LowerTriangular(one(SMatrix{1,1,T}))
end

function SequentialMonteCarlo.simulate_observation_noise(x, ::StochasticVolatility{T}, t, data, θ) where {T}
    return randn(SVector{1,T})
end
