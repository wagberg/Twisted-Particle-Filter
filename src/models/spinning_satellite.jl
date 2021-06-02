"""
A non-linear state-space model with additive Gaussian noise.

xₜ₊₁ = f(xₜ, uₜ, vₜ) = xₜ + δt
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

function SequentialMonteCarlo.transition_function(x, ::SpinningSatellite, t::Integer, data, θ::SpinningSatelliteParameter)
    @unpack dt, J, Q = θ
    x .+ dt*data.u[t]./J .+ [dt*(J[2]-J[3])/J[1]*x[2]*x[3],dt*(J[3]-J[1])/J[2]*x[3]*x[1],dt*(J[1]-J[2])/J[3]*x[1]*x[2]]
end

function SequentialMonteCarlo.observation_function(x, ::SpinningSatellite, t::Integer, data, θ::SpinningSatelliteParameter)
    @unpack c = θ
    clamp.(x, -c, c)
end
