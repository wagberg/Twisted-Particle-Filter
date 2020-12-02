"""
A linear Gaussian state-space model parameterized by the state dimension
"""

mutable struct SVParticle <: Particle
    x::Float64
    SVParticle() = new(0)
end

function copy!(dest::SVParticle, src::SVParticle)
    dest.x = src.x
end

"""
Convert the particle to a static vector.
This is used to compute gradients of the transition function.
"""
function SVector(p::SVParticle)
    SVector{1, Float64}(p.x);
end

function statenames(::SVParticle)
    Symbol.(["x"])
end

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
struct SV <: SSM{SVParticle}
end

@with_kw struct SVParameter
    μ₁::Float64    = randn()
    logσ₁::Float64 = randn()
    logϕ::Float64  = randn()
    logσ::Float64  = randn()
    logβ::Float64  = randn() 
end

function simulate_initial!(p::SVParticle, ::SV, data, θ)
    @unpack μ₁, logσ₁ = θ
    p.x = rand(Normal(μ₁, exp(logσ₁)))
    nothing
end

function log_initial_density(p::SVParticle, ::SV, data, θ)
    @unpack μ₁, logσ₁ = θ
    logpdf(Normal(μ₁, exp(logσ₁)), p.x)
end

function simulate_observation!(y::AVec{<:AFloat}, p::SVParticle, ::SV, t::Integer, data, θ)
    @unpack logβ = θ
    y.= rand(Normal(0.0, exp(logβ).*exp.(p.x)))
    nothing
end

function log_observation_density!(p::SVParticle, ::SV, t::Integer, data, θ)
    @unpack logβ = θ
    logpdf(Normal(0.0, exp(logβ).*exp.(p.x)), data.y[t][1])
end

function simulate_transition!(pnext::SVParticle, pcurr::SVParticle, ::SV, t::Integer, data, θ)
    @unpack logϕ, logσ = θ
    pnext.x = rand(Normal(exp(logϕ)*pcurr.x, exp(logσ)))
    nothing
end

function log_transition_density(pnext::SVParticle, pcurr::SVParticle, ::SV, t::Integer, data, θ)
    @unpack logϕ, logσ = θ
    logpdf(Normal(pnext.x - exp(logϕ)*pcurr.x, exp(ogσ)))
end

function transition_function(x, ::SV, t::Integer, data, θ)
    @unpack logϕ, logσ = θ
    exp(logϕ)*x
end

function observation_function(x, ::SV, t::Integer, data, θ)
    return 0
end

function transition_covariance(x, ::SV, t::Integer, data, θ)
    @unpack logσ = θ
    SMatrix{1,1}(exp(2*logσ))
end

function observation_covariance(x, ::SV, t::Integer, data, θ)
    @unpack logβ = θ
    SMatrix{1,1}(exp(2*logβ + x))
end


