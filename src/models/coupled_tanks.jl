const CoupledTankParticle = FloatParticle{2}

"""
Coupled tank model
"""
struct CoupledTank <: SSM{CoupledTankParticle}
end

@with_kw struct TankParameter
    μ0::SVector{2} = [6.225503938785832, 4.9728]
    Σ0::Symmetric = Symmetric(Matrix{Float64}(I,2,2))
    k₁::Float64 = 0.054115135099972
    k₂::Float64 = 0.069031758496706
    k₃::Float64 = 0.044688311520571
    k₄::Float64 = 0.224558186566615
    k₅::Float64 = 0.002262430348693
    k₆::Float64 = 0.006535014001622
    σw::Float64 = sqrt(0.002513535972694)
    σe::Float64 = sqrt(0.013531902670792)
    Ts::Float64 = 4.0
end

function SequentialMonteCarlo.transition_function(x, ::CoupledTank, t, data, θ)
    @unpack k₁, k₂, k₃, k₄, k₅, k₆, Ts  = θ
    xp = clamp.(x, 0.0, 10.0)
    @inbounds xp[1] += Ts*(-k₁*sqrt(xp[1]) - k₅*xp[1] + k₃*data.u[t][1])
    @inbounds xp[2] += Ts*(k₁*sqrt(xp[1]) + k₅*xp[1] - k₂*sqrt(xp[2]) - k₆*xp[2] + k₄*max.(x[1]-10.0, 0.0))
    return xp
end

function SequentialMonteCarlo.observation_function(x, ::CoupledTank, t, data, θ)
    clamp.(x[2:2], 0.0, 10.0)
end

function SequentialMonteCarlo.transition_covariance(x, ::CoupledTank, t, data, θ)
    @unpack σw = θ
    σw^2 * I(2)
end

function SequentialMonteCarlo.observation_covariance(x, ::CoupledTank, t, data, θ)
    @unpack σe = θ
    σe^2* I(1)
end