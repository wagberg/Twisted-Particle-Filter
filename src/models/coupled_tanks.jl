using Parameters
using LinearAlgebra
using Distributions
using PDMats
# using SequentialMonteCarlo: transition_function, observation_function
import SequentialMonteCarlo: transition_function, observation_function, initial_density,
    transition_noise, observation_noise
import Base: clamp

const CoupledTankParticle{T} = FloatParticle{2, T}

"""
Coupled tank model
"""
struct CoupledTank{T} <: AbstractSSM{CoupledTankParticle{T}}
end
CoupledTank() = CoupledTank{Float64}()

@with_kw struct TankParameter{QT,RT,μT,ΣT}
    # initial_density::IT = MvNormal(SVector{2, Float64}([6.225503938785832, 4.9728]), SMatrix{2,2,Float64}(I))
    # process_noise::PT = MvNormal(zero(SVector{2, Float64}), SMatrix{2,2,Float64}(0.002513535972694*I))
    # observation_noise::OT = MvNormal(zero(SVector{1, Float64}), SMatrix{1,1,Float64}(0.013531902670792*I))
    # μ0::SVector{2,T} = SVector{2, T}([6.225503938785832, 4.9728])
    # Σ0::SMatrix{2,2,T} = SMatrix{2,2,T}(I)
    k₁::Float64 = 0.054115135099972
    k₂::Float64 = 0.069031758496706
    k₃::Float64 = 0.044688311520571
    k₄::Float64 = 0.224558186566615
    k₅::Float64 = -0.002262430348693
    k₆::Float64 = -0.006535014001622
    # σw²::Float64 = 0.002513535972694
    Q::QT = PDMat(SMatrix{2,2}(0.002513535972694*I))
    # σe²::Float64 = 0.013531902670792
    R ::RT= PDMat(SMatrix{1,1}(0.013531902670792*I))
    μ₀::μT = SVector{2}([6.225503938785832, 5])
    Σ₀::ΣT = PDMat(SMatrix{2,2}(0.1*I))
    # ξ::FLoat64 = 6.225503938785832 # Initial height in upper tank
    Ts::Float64 = 4.0
end

initial_density(::CoupledTank, data, θ) = MvNormal(θ.μ₀, θ.Σ₀)
transition_noise(xₜ, ::CoupledTank, t, data, θ) = MvNormal(θ.Q)
observation_noise(xₜ, ::CoupledTank, t, data, θ) = MvNormal(θ.R)

# k₁:  0.054115135099972
# k2:  0.069031758496706
# k₃:  0.044688311520571
# k₄:  0.224558186566615
# k₅: -0.002262430348693
# k6: -0.006535014001622
# σw²: 0.002513535972694
# σe²: 0.013531902670792
# ξ:   6.225503938785832

clamp(x::StaticVector, a, b) = max.(min.(b, x),a)

function transition_function(x, v, ::CoupledTank, t, data, θ)
    @unpack k₁, k₂, k₃, k₄, k₅, k₆, Ts  = θ
    xp = clamp(SVector(x), 0.0, 10.0)
    sqrtxp = sqrt.(xp)
    K₁ = Ts*@SMatrix [-k₁ 0; k₁ -k₂]
    K₂ = Ts*@SMatrix [-k₅ 0; k₅ -k₆]
    K₃ = Ts*@SMatrix [k₃ ; 0]
    K₄ = Ts*@SMatrix [0 0 ; k₄ 0]
    return xp + K₁*sqrtxp + K₂*xp + K₃*data.u[t] + K₄*(x.-xp) + v
end

function observation_function(x, e, ::CoupledTank, t, data, θ)
    C = @SMatrix [0.0 1.0]
    return clamp(C*SVector(x), 0.0, 10.0) + e
end
