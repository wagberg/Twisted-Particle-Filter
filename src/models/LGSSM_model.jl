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
    LogInitPotFun <: Function, LogPotFun <: Function} <: SSM
    Mi!::InitSampler # Function sampling from the initial distribution of particles.
    lMi::LogInitDens # Function returning log density of the initial distribution.
    M!::ForwardSampler # Function sampling from the propagating model.
    lM::LogForwardDens # Function returning log density of the propagating model.
    lGi::LogInitPotFun # Potential function returning log-weight for a particle, t = 1.
    lG::LogPotFun # Potential function returning log-weight for a particle, t >= 2
    function GenericSSM(::Type{P}, Mi!::Function, lMi::Function,
            M!::Function, lM::Function,
            lGi::Function, lG::Function) where {P <: Particle}
        new{P, typeof(Mi!), typeof(lMi), typeof(M!),
        typeof(lM), typeof(lGi), typeof(lG)}(Mi!, lMi, M!, lM, lGi, lG);
    end
end

LGSSM = let 
    function f(pcurr <:LGSSMParticle, t::Integer, data, θ)
        @unpack A, B, N = θ
        A*pcurr.x + B*data.u[t] + N * w
    end
    function h(p <: LGSSMParticle
end