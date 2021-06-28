# using Parameters
using UnPack
using LinearAlgebra
using Distributions
using PDMats
using StaticArrays
using QuickTypes
using SequentialMonteCarlo
import SequentialMonteCarlo: transition_function, observation_function, initial_density,
    transition_noise, observation_noise, toSVector, par_to_vec, vec_to_par
import Base: clamp

const CoupledTankParticle{T} = FloatParticle{2, T}

"""
Coupled tank model
"""
struct CoupledTank{T} <: AbstractSSM{CoupledTankParticle{T}}
end
CoupledTank() = CoupledTank{Float64}()

# @with_kw struct TankParameter{T}
#     k = SA[0.054115135099972, 0.069031758496706, 0.044688311520571, 0.224558186566615, -0.002262430348693, -0.006535014001622]
#     Q::PDMat{T,SMatrix{2,2,T,4}} = PDMat(SMatrix{2,2}(0.002513535972694*I))
#     R::PDMat{T,SMatrix{1,1,T,1}} = PDMat(SMatrix{1,1}(0.013531902670792*I))
#     μ::SVector{2, T} = SVector{2}([6.225503938785832, 5])
#     Σ::PDMat{T,SMatrix{2,2,T,4}} = PDMat(SMatrix{2,2}(0.1*I))
#     Ts::Float64 = 4.0
# end

@qstruct_fp TankParameter(;
    k = SA[0.054115135099972, 0.069031758496706, 0.044688311520571, 0.224558186566615, -0.002262430348693, -0.006535014001622],
    Q = PDMat(SMatrix{2,2}(0.002513535972694*I)),
    R = PDMat(SMatrix{1,1}(0.013531902670792*I)),
    μ = SVector{2}([6.225503938785832, 5]),
    Σ = PDMat(SMatrix{2,2}(0.1*I)),
    Ts = 4.0
    )

initial_density(::CoupledTank, data, θ::TankParameter) = MvNormal(θ.μ, θ.Σ)
# initial_density(::CoupledTank, data, θ::AbstractVector) = MvNormal(SVector{2}(θ[7:8]), PDMat(SMatrix{2,2}(θ[11]*I)))
initial_density(::CoupledTank, data, θ::AbstractVector{T}) where T =
    @views MvNormal(SVector{2,T}(θ[7:8]), PDMat(SMatrix{2,2,T,4}(θ[11]*I), Cholesky(SMatrix{2,2,T,4}(sqrt(θ[11]*I)),'U',0)))

transition_noise(xₜ, ::CoupledTank, t, data, θ::TankParameter) = MvNormal(θ.Q)
transition_noise(xₜ, ::CoupledTank, t, data, θ::AbstractVector) = MvNormal(PDMat(SMatrix{2,2}(θ[9]*I)))
observation_noise(xₜ, ::CoupledTank, t, data, θ::TankParameter) = MvNormal(θ.R)
observation_noise(xₜ, ::CoupledTank, t, data, θ::AbstractVector) = MvNormal(PDMat(SMatrix{1,1}(θ[10]*I)))

function par_to_vec(θ::TankParameter)
    @unpack k, Q, R, μ, Σ = θ
    return SVector{11}([k..., μ..., Q.mat[1], R.mat[1], Σ.mat[1]])
end

function vec_to_par(::CoupledTank, x)
    @assert length(x) == 11
    Q = PDMat(SMatrix{2,2}(x[9]*I))
    R = PDMat(SMatrix{1,1}(x[10]*I))
    Σ = PDMat(SMatrix{2,2}(x[11]*I))
    return TankParameter(k=SVector{6}(x[1:6]),Q=Q,R=R,μ=SVector{2}(x[7:8]),Σ=Σ)
end

# function vec_to_par(::CoupledTank, x)
#     @assert length(x) == 11
#     Q = PDMat(SMatrix{2,2}(x[9]*I))
#     R = PDMat(SMatrix{1,1}(x[10]*I))
#     Σ = PDMat(SMatrix{2,2}(x[11]*I))
#     return TankParameter(x[1:6],Q,R,x[7:8],Σ)
# end


clamp(x::StaticVector, a, b) = max.(min.(b, x),a)

function transition_function(x, v, ::CoupledTank, t, data, θ::AbstractVector)
    k₁, k₂, k₃, k₄, k₅, k₆ = θ
    xp = clamp(x, eps(Float64), 10.0)
    sqrtxp = sqrt.(xp)
    K₁ = Ts*(@SMatrix  [-k₁ 0; k₁ -k₂])
    K₂ = Ts*(@SMatrix [-k₅ 0; k₅ -k₆])
    K₃ = Ts*(@SMatrix [k₃ ; 0])
    K₄ = Ts*(@SMatrix [0 0 ; k₄ 0])
    return xp + K₁*sqrtxp + K₂*xp + K₃*data.u[t] + K₄*(x.-xp) + v
end

function transition_function(x, v, ::CoupledTank, t, data, θ::TankParameter)
    @unpack k, Ts = θ
    k₁, k₂, k₃, k₄, k₅, k₆ = k
    xp = clamp(x, eps(Float64), 10.0)
    sqrtxp = sqrt.(xp)
    K₁ = Ts*(@SMatrix  [-k₁ 0; k₁ -k₂])
    K₂ = Ts*(@SMatrix [-k₅ 0; k₅ -k₆])
    K₃ = Ts*(@SMatrix [k₃ ; 0])
    K₄ = Ts*(@SMatrix [0 0 ; k₄ 0])
    return xp + K₁*sqrtxp + K₂*xp + K₃*data.u[t] + K₄*(x.-xp) + v
end

function observation_function(x, e, ::CoupledTank, t, data, θ)
    C = @SMatrix [0.0 1.0]
    return clamp(C*x, eps(Float64), 10.0) + e
end

# @with_kw struct TankParameter{QT,RT,μT,ΣT}
#     # initial_density::IT = MvNormal(SVector{2, Float64}([6.225503938785832, 4.9728]), SMatrix{2,2,Float64}(I))
#     # process_noise::PT = MvNormal(zero(SVector{2, Float64}), SMatrix{2,2,Float64}(0.002513535972694*I))
#     # observation_noise::OT = MvNormal(zero(SVector{1, Float64}), SMatrix{1,1,Float64}(0.013531902670792*I))
#     # μ0::SVector{2,T} = SVector{2, T}([6.225503938785832, 4.9728])
#     # Σ0::SMatrix{2,2,T} = SMatrix{2,2,T}(I)
#     k₁::Float64 = 0.054115135099972
#     k₂::Float64 = 0.069031758496706
#     k₃::Float64 = 0.044688311520571
#     k₄::Float64 = 0.224558186566615
#     k₅::Float64 = -0.002262430348693
#     k₆::Float64 = -0.006535014001622
#     # σw²::Float64 = 0.002513535972694
#     Q::QT = PDMat(SMatrix{2,2}(0.002513535972694*I))
#     # σe²::Float64 = 0.013531902670792
#     R ::RT= PDMat(SMatrix{1,1}(0.013531902670792*I))
#     μ₀::μT = SVector{2}([6.225503938785832, 5])
#     Σ₀::ΣT = PDMat(SMatrix{2,2}(0.1*I))
#     # ξ::FLoat64 = 6.225503938785832 # Initial height in upper tank
#     Ts::Float64 = 4.0
# end

# function toSVector(θ::TankParameter)
#     @unpack k₁, k₂, k₃, k₄, k₅, k₆, Q, R = θ
#     return @SVector [k₁, k₂, k₃, k₄, k₅, k₆, Q.mat[1], R.mat[1]]
# end

# function TankParameter(x::AbstractVector)
#     @assert length(x) == 8
#     k₁, k₂, k₃, k₄, k₅, k₆, σᵥ, σₑ = x
#     Q = PDMat(SMatrix{2,2}(σᵥ*I))
#     R = PDMat(SMatrix{1,1}(σₑ*I))
#     return TankParameter(k₁=k₁, k₂=k₂, k₃=k₃, k₄=k₄, k₅=k₅, k₆=k₆, Q=Q, R=R)
# end

# k₁:  0.054115135099972
# k2:  0.069031758496706
# k₃:  0.044688311520571
# k₄:  0.224558186566615
# k₅: -0.002262430348693
# k6: -0.006535014001622
# σw²: 0.002513535972694
# σe²: 0.013531902670792
# ξ:   6.225503938785832



# function transition_function(x, v, ::CoupledTank, t, data, θ)
#     @unpack k₁, k₂, k₃, k₄, k₅, k₆, Ts  = θ
#     xp = clamp(x, eps(Float64), 10.0)
#     sqrtxp = sqrt.(xp)
#     K₁ = Ts*@SMatrix [-k₁ 0; k₁ -k₂]
#     K₂ = Ts*@SMatrix [-k₅ 0; k₅ -k₆]
#     K₃ = Ts*@SMatrix [k₃ ; 0]
#     K₄ = Ts*@SMatrix [0 0 ; k₄ 0]
#     return xp + K₁*sqrtxp + K₂*xp + K₃*data.u[t] + K₄*(x.-xp) + v
# end

# function observation_function(x, e, ::CoupledTank, t, data, θ)
#     C = @SMatrix [0.0 1.0]
#     return clamp(C*x, eps(Float64), 10.0) + e
# end
