"""
A non-linear state-space model with additive Gaussian noise.
"""
struct SpinningSatellite <: SSM{FloatParticle{3}}
end

@with_kw struct SpinningSatelliteParameter
    dt::Float64    = 0.001
    c::Float64     = 10.0
    J::SVector{3}  = [1; 5; 5]
    Q::Symmetric   = Symmetric(0.001.*Matrix{Float64}(I,3,3))
    R::Symmetric   = Symmetric(0.3.*Matrix{Float64}(I,3,3))
    μ0::SVector{3} = [10.0; 0.0; 0.0]
    Σ0::Symmetric  = Symmetric(Diagonal([1.0; 1.0; 1.0]))
end

# function simulate_initial!(p::FloatParticle{3}, ::SpinningSatellite, data, θ::SpinningSatelliteParameter)
#     @unpack μ0, Σ0 = θ
#     p.x .= rand(MvNormal(μ0, Σ0))
#     nothing
# end

# function log_initial_density(p::FloatParticle{3}, ::SpinningSatellite, data, θ::SpinningSatelliteParameter)
#     @unpack μ0, Σ0 = θ
#     logpdf(MvNormal(μ0, Σ0), p.x )
# end

function SequentialMonteCarlo.transition_function(x, ::SpinningSatellite, t::Integer, data, θ::SpinningSatelliteParameter)
    @unpack dt, J, Q = θ
    x .+ dt*data.u[t]./J .+ [dt*(J[2]-J[3])/J[1]*x[2]*x[3],dt*(J[3]-J[1])/J[2]*x[3]*x[1],dt*(J[1]-J[2])/J[3]*x[1]*x[2]]
end

function SequentialMonteCarlo.observation_function(x, ::SpinningSatellite, t::Integer, data, θ::SpinningSatelliteParameter)
    @unpack c = θ
    clamp.(x, -c, c)
end

# function initial_mean(::SpinningSatellite, data, θ::SpinningSatelliteParameter)
#     @unpack μ0 = θ
#     μ0
# end

# function initial_covariance(::SpinningSatellite, data, θ::SpinningSatelliteParameter)
#     @unpack Σ0 = θ
#     Σ0
# end

# function transition_covariance(x, ::SpinningSatellite, t::Integer, data, θ::SpinningSatelliteParameter)
#     @unpack Q = θ
#     Q
# end

# function observation_covariance(x, ::SpinningSatellite, t::Integer, data, θ::SpinningSatelliteParameter)
#     @unpack R = θ
#     R
# end
