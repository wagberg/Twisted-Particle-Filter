
using SequentialMonteCarlo
import SequentialMonteCarlo: transition_function, observation_function, initial_density,
    transition_noise, observation_noise
using Distributions
using LinearAlgebra
using PDMats
using Parameters

# x(t+1) = 0.5x + 25x/(1+x^2) + 8cos(1.2(t-1))
# y(t) = 0.05x^2

"""
Coupled tank model
"""
struct StandardBenchmark{T} <: AbstractSSM{FloatParticle{1,T}} end
StandardBenchmark() = StandardBenchmark{Float64}()

@with_kw struct StandardBenchmarkPar{T,QT,RT,IT}
    k₁::T = 0.5
    k₂::T = 25.0
    k₃::T = 8.0
    k₄::T = 1.0/20.0
    # Σᵥ::PDMat{T,<:SMatrix{1,1,T}} = PDMat(SMatrix{1,1}(10*I)) # transition noise std
    Σᵥ::QT = PDMat(SMatrix{1,1}(10*I)) # transition noise std
    # Σₑ::PDMat{T,<:SMatrix{1,1,T}} = PDMat(SMatrix{1,1}(1*I)) # measurement noise std
    Σₑ::RT = PDMat(SMatrix{1,1}(1*I)) # measurement noise std
    μ₀::SVector{1,T} = SVector{1}(5.0) # initial mean
    # Σ₀::PDMat{T,<:SMatrix{1,1,T}} = PDMat(SMatrix{1,1}(5*I))
    Σ₀::IT = PDMat(SMatrix{1,1}(5*I))
end

@inline initial_density(::StandardBenchmark, data, θ) = MvNormal(θ.μ₀, θ.Σ₀)
@inline transition_noise(xₜ, ::StandardBenchmark, t, data, θ) = MvNormal(θ.Σᵥ)
@inline observation_noise(xₜ, ::StandardBenchmark, t, data, θ) = MvNormal(θ.Σₑ)

function transition_function(xₜ, vₜ, ::StandardBenchmark, t, data, θ)
    @unpack k₁, k₂, k₃ = θ
    return k₁*xₜ + k₂*xₜ./(1 .+ xₜ.*xₜ) + k₃*SVector{1}(cos(t)) + vₜ
end

function observation_function(xₜ, eₜ, ::StandardBenchmark, t, data, θ)
    @unpack k₄ = θ
    return k₄.*xₜ.*xₜ + eₜ
end
