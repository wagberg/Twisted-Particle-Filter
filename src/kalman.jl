using StateSpace
using LinearAlgebra
using Distributions
using Random
import Base

abstract type Particle end

function statenames(::Type{P}) where P <: Particle
    fieldnames(P);
end

struct KalmanStorage{P <: Particle,T <: AbstractFloat} <: SSMStorage
    M::Vector{P}
    Σ::Array{T,3}
    filtered_index::Base.RefValue{Int}
  
    function ParticleStorage(::Type{P}, ts_length::Integer) where {P <: Particle}
        M = [P() for i in 1:ts_length];
        Σ = zeros(Float64, length(M[1]), length(M[1]), ts_length);
        filtered_index = Ref(0);
        new{P,Float64}(M, Σ,filtered_index);
    end
end


mutable struct LGSSMParticle{N} <: Particle
    x::MVector{N, Float64}
end

function copy!(dest::LGSSMParticle{N}, src::LGSSMParticle{N}) where N
    for i in eachindex(dest.x)
        @inbounds dext.x[i] = src.x[i]
    end
    dest;
end

function SVector{N, Float64}(p::LGSSMParticle{N}) where N
    SVector{N, Float64}(p.x);
end

function statenames(::Type{FloatParticle{N}}) where N
    Symbol.("x" .* string.(collect(1:N)));
end

struct GenericSSM{P <: Particle, InitSampler <: Function, LogInitDens <: Function,
    ForwardSampler <: Function, LogForwardDens <: Function,
    LogInitPotFun <: Function, LogPotFun <: Function, Predict <: Function,
    Measure <: Function} <: SSM
    Qi!::InitProposal # Function sampling from the initial distribution of particles.
    lQi::LogInitDens # Function returning log density of the initial distribution.
    M!::ForwardSampler # Function sampling from the propagating model.
    lM::LogForwardDens # Function returning log density of the propagating model.
    lGi::LogInitPotFun # Potential function returning log-weight for a particle, t = 1.
    lG::LogPotFun # Potential function returning log-weight for a particle, t >= 2
    predict::Predict
    measure::Measure
    function GenericSSM(::Type{P}, Mi!::Function, lMi::Function,
            M!::Function, lM::Function,
            lGi::Function, lG::Function,
            f::Function, h::Function) where {P <: Particle}
        new{P, typeof(Mi!), typeof(lMi), typeof(M!),
        typeof(lM), typeof(lGi), typeof(lG),
        typeof(f), typeof(h)}(Mi!, lMi, M!, lM, lGi, lG, f, h);
    end
end

function build LGSSM(StateDim::Integer)
    LGSSM = let 
        function f(p::LGSSMParticle, w <: AbstractVector, t::Integer, data, θ)
            @unpack A, B, N = θ
            A*pcurr.x .+ B*data.u[t] .+ N * w
        end
        function h(p::LGSSMParticle, e <: AbstractVector, t::Integer, data, θ)
            @unpack C, D, M = θ
            C*p.x .+ D*data.u[t] .+ M*e
        end
        function Mi!(p::LGSSMParticle, data, θ)
            @unpack μ0, Σ0 = θ
            p.x = rand(MvNormal(μ0, Σ0))
        end 
        function lMi(p::LGSSMParticle, data, θ)
            @unpack μ0, Σ0 = θ
            logpdf(MvNormal(μ0, Σ0), p.x)
        end
        function M!(pnext::LGSSMParticle, pcurr::LGSSMParticle, t::Int, data, θ)
            @unpack Q = θ
            w = rand(MvNormal(Q))
            pnext.x = f(pcurr, w, t, data, θ)
        end
        function lM(pnext::LGSSMParticle, pcurr::LGSSMParticle, t::Int, data, θ)
        end
        GenericSSM()
    end
end