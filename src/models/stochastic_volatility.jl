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





function SequentialMonteCarlo.simulate_initial!(p::StochasticVolatilityParticle, ::StochasticVolatility, data, θ)
    @unpack μ₁, logσ₁ = θ
    p.x .= rand(Normal(μ₁, exp(logσ₁)))
    nothing
end

function SequentialMonteCarlo.log_initial_density(p::StochasticVolatilityParticle, ::StochasticVolatility, data, θ)
    @unpack μ₁, logσ₁ = θ
    logpdf(Normal(μ₁, exp(logσ₁)), p.x)
end

function simulate_observation!(y::AbstractVector{<:AbstractFloat}, p::StochasticVolatilityParticle, ::StochasticVolatility, t::Integer, data, θ)
    @unpack logβ = θ
    y.= rand(Normal(0.0, exp(logβ).*exp.(p.x)))
    nothing
end

function log_observation_density!(p::StochasticVolatilityParticle, ::StochasticVolatility, t::Integer, data, θ)
    @unpack logβ = θ
    logpdf(Normal(0.0, exp(logβ).*exp.(p.x)), data.y[t][1])
end

function simulate_transition!(pnext::StochasticVolatilityParticle, pcurr::StochasticVolatilityParticle, ::StochasticVolatility, t::Integer, data, θ)
    @unpack logϕ, logσ = θ
    pnext.x = rand(Normal(exp(logϕ)*pcurr.x, exp(logσ)))
    nothing
end

function log_transition_density(pnext::StochasticVolatilityParticle, pcurr::StochasticVolatilityParticle, ::StochasticVolatility, t::Integer, data, θ)
    @unpack logϕ, logσ = θ
    logpdf(Normal(pnext.x - exp(logϕ)*pcurr.x, exp(ogσ)))
end



function transition_covariance(x, ::StochasticVolatility, t::Integer, data, θ)
    @unpack logσ = θ
    SMatrix{1,1}(exp(2*logσ))
end

function observation_covariance(x, ::StochasticVolatility, t::Integer, data, θ)
    @unpack logβ = θ
    SMatrix{1,1}(exp(2*logβ + x))
end


