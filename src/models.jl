
using StaticArrays
using LinearAlgebra
using Distributions
using Parameters
using ForwardDiff, DiffResults
include("src/resampling.jl")

const AVec = AbstractVector;
const AMat = AbstractMatrix;
const AFloat = AbstractFloat;

"""
Abstract Particle type
"""
abstract type Particle end

"""
FloatParticle{N}
"""
struct FloatParticle{N} <: Particle
    x::MVector{N,Float64}
end

function FloatParticle{N}() where {N}
    FloatParticle(zero(MVector{N,Float64}));
end

function FloatParticle(x::AVec)
    FloatParticle{length(x)}(x)
end

function SVector{N,Float64}(p::FloatParticle{N}) where N
    SVector{N,Float64}(p.x);
end

SVector(p::FloatParticle{N}) where N = SVector{N,Float64}(p)

# import Base.convert
# convert(::Type{T}, p::FloatParticle{N}) where {T<:StaticArray, N} = T(p)

function copy!(dest::FloatParticle{N}, src::FloatParticle{N}) where N
    for i in eachindex(dest.x)
        @inbounds dest.x[i] = src.x[i];
    end
    dest;
end

abstract type AbstractSSM{P <: Particle} end
abstract type AbstractGenericSSM{P} <: AbstractSSM{P} end
abstract type AbstractFunctionalSSM{P} <: AbstractGenericSSM{P} end

struct GenericSSM{P <: Particle,InitSampler <: Function,LogInitDens <: Function,ForwardSampler <: Function,LogForwardDens <: Function,LogInitPotFun <: Function,LogPotFun <: Function} <: AbstractGenericSSM{P}
end

struct FunctionalSSM{P <: Particle,InitSampler <: Function,LogInitDens <: Function,TransitionSampler <: Function,LogTransitionDens <: Function,DataSampler <: Function,LogDataDensity <: Function} <: AbstractGenericSSM{P}
    initial!::InitSampler
    initial_logpdf::LogInitDens
    transition!::TransitionSampler
    transition_logpdf::LogTransitionDens
    observe!::DataSampler
    observe_logpdf::LogDataDensity
    function FunctionalSSM(::Type{P},
        initial!::Function, initial_logpdf::Function,
        transition!::Function, transition_logpdf::Function,
        observe!::Function, observe_logpdf::Function) where P <: Particle
        new{P,typeof(initial!),typeof(initial_logpdf),typeof(transition!),typeof(transition_logpdf),typeof(observe!),typeof(observe_logpdf)}(initial!, initial_logpdf, transition!, transition_logpdf, observe!, observe_logpdf)
    end
end

struct Data
    u::AVec{AVec{<:AFloat}}
end

function build_LGSSM(N::Int)
    LGSSM = let
        function initial!(p::FloatParticle{N}, data, θ) where N
            @unpack μ0, Σ0 = θ
            p.x .= rand(MvNormal(μ0, Symmetric(Σ0)))
            nothing
        end
        function initial_logpdf(p::FloatParticle{N}, data, θ) where N
            @unpack μ0, Σ0 = θ
            logpdf(MvNormal(μ0, Symmetric(Σ0)), p.x)
        end
        function transition!(pnext::FloatParticle{N}, pcurr::FloatParticle{N}, t::Int, data, θ) where N
            @unpack A, B, Q = θ
            pnext.x .= A * pcurr.x .+ B * data.u[t] .+ rand(MvNormal(Symmetric(Q)))
            nothing
        end
        function transition_logpdf(pnext::FloatParticle{N}, pcurr::FloatParticle{N}, t::Int, data, θ) where N
            @unpack A, B, Q = θ
            logpdf(MvNormal(Symmetric(Q)), pnext.x .- A * pcurr.x .- B * data.u[t])
        end
        function observe!(y::AVec{<:AFloat}, p::FloatParticle{N}, t::Int, data, θ) where N
            @unpack C, D, R = θ
            y .= C * p.x .+ D * data.u[t] .+ rand(MvNormal(Symmetric(R)))
            nothing
        end
        function observe_logpdf(y::AVec{<:AFloat}, p::FloatParticle{N}, t::Int, data, θ) where N
            @unpack C, D, R = θ
            logpdf(MvNormal(Symmetric(R)), y .- C * p.x .- D * data.u[t])
        end
        function state_jacobian()
        end
        function noise_jacobian()
        end
        FunctionalSSM(FloatParticle{N}, initial!, initial_logpdf, transition!, transition_logpdf, observe!, observe_logpdf)
    end
end


function simulate!(y::AVec{<: AVec{<: Real}}, model::AbstractGenericSSM{P}, data, θ) where P <: Particle
    @assert !isempty(y) "the vector `y` must not be empty.";
    N = length(y);
    pcur = P(); pnext = P();

    model.initial!(pcur, data, θ);
    model.observe!(y[1], pcur, 1, data, θ);
    for i in 2:N
        model.transition!(pnext, pcur, i, data, θ);
        model.observe!(y[i], pnext, i, data, θ);
        copy!(pcur, pnext);
    end
    nothing;
end

"""
Simulate both the observations and the latent state.
"""
function simulate!(y::AVec{<: AVec{<: Real}},
                   x::AVec{P}, model::AbstractGenericSSM{P}, data, θ) where {P <: Particle}
    @assert !isempty(y) "the vector `y` must not be empty.";
    @assert !isempty(x) "the vector `x` must not be empty.";
    N = length(y);
    @assert length(x) == N "`the lengths of `x` and `y` do not match.";

    model.initial!(x[1], data, θ);
    model.observe!(y[1], x[1], 1, data, θ);
    for i in 2:N
        model.transition!(x[i], x[i - 1], i - 1, data, θ);
        model.observe!(y[i], x[i], i, data, θ);
    end
    nothing;
end

abstract type SSMStorage end

struct SSMInstance{M <: AbstractSSM, S <: SSMStorage, D <: Any}
    model::M
    storage::S
    data::D
    len::Int
    function SSMInstance(m::AbstractSSM, s::SSMStorage, d, len::Integer)
      @assert typeof(s) <: storagetype(m)
      @assert len > 0 "the length of the timeseries must be > 0.";
      new{typeof(m), typeof(s), typeof(d)}(m, s, d, len);
    end
end

import Base.length
function length(ssm::SSMInstance)
  ssm.len;
end

struct ParticleStorage{P <: Particle, T <: AFloat} <: SSMStorage
    X::Matrix{P}
    W::Matrix{T}
    A::Matrix{Int}
    V::Vector{T}
    wnorm::Vector{T}
    ref::Vector{Int}
    filtered_index::Base.RefValue{Int}
  
    function ParticleStorage(::Type{P}, npar::Integer,
                             ts_length::Integer) where {P <: Particle}
      X = [P() for i in 1:npar, j in 1:ts_length];
      W = zeros(Float64, npar, ts_length);
      V = zeros(Float64, npar);
      wnorm = zeros(Float64, npar);
      A = zeros(typeof(npar), npar, ts_length); # The additional last column is used in BS.
      ref = ones(typeof(npar), ts_length);
      filtered_index = Ref(0);
      new{P, Float64}(X, W, A, V, wnorm, ref, filtered_index);
    end
end

capacity(ps::ParticleStorage) = size(ps.X);
capacity(ps::ParticleStorage, dim::Integer) = size(ps.X, dim);
particle_count(ps::ParticleStorage) = capacity(ps, 1);
particle_dimension(ps::ParticleStorage) = length(ps.X[1, 1]);

function _init!(ps::ParticleStorage)
    ps.filtered_index[] = 0;
    ps.V .= 0.0;
    nothing
end

function storagetype(model::AbstractGenericSSM)
    ParticleStorage;
end

"""
Normalise a vector of weight logarithms, `log_weights`, in place.
After normalisation, the weights are in the linear scale.
Additionally, the logarithm of the linear scale mean weight is returned.
"""
@inline function normalise_logweights!(log_weights::AVec{<: Real})
  m = maximum(log_weights);
  if isapprox(m, -Inf) # To avoid NaN in case that all values are -Inf.
    log_weights .= zero(eltype(log_weights));
    return -Inf;
  end
  log_weights .= exp.(log_weights .- m);
  log_mean_weight = m + log(mean(log_weights));
  normalize!(log_weights, 1);
  log_mean_weight;
end

"""
Compute log(sum(exp.(`x`))) in a numerically stable way.
"""
@inline function logsumexp(x::AbstractArray{<: Real})
  m = maximum(x);
  isapprox(m, -Inf) && (return -Inf;) # If m is -Inf, without this we would return NaN.
  s = 0.0;
  for i in eachindex(x)
    @inbounds s += exp(x[i] - m);
  end
  m + log(s);
end

##
function bpf!(ssm::SSMInstance{<: AbstractGenericSSM}, θ;
    resampling::Resampling=MultinomialResampling())
    model = ssm.model; ps = ssm.storage; data = ssm.data;

    # Initialisation.
    _init!(ps);
    X = ps.X; W = ps.W; A = ps.A; V = ps.V; ref = ps.ref;
    wnorm = ps.wnorm;
    ts_length = length(ssm);
    n_particles = particle_count(ps);
    all_particle_indices = Base.OneTo(n_particles);

    # Simulate from initial distribution.
    for j in all_particle_indices
        @inbounds model.initial!(X[j, 1], data, θ);
    end

    # Compute initial weights.
    for j in all_particle_indices
        @inbounds W[j, 1] = model.initial_logpdf(X[j, 1], data, θ);
        @inbounds wnorm[j] = W[j, 1];
    end
    V .= V .+ normalise_logweights!(wnorm);
    ps.filtered_index[] += 1;

    for t in 2:ts_length
        a = view(A, :, t - 1);
    # Resample and propagate surviving particles.
       
        resample!(a, wnorm, resampling);
        for j in all_particle_indices
            @inbounds model.transition!(X[j, t], X[a[j], t - 1], t, data, θ);
        end
    # Compute weights.  
        for j in all_particle_indices
            @inbounds W[j, t] = model.transition_logpdf(X[a[j], t - 1], X[j, t], t, data, θ);
            @inbounds wnorm[j] = W[j, t];
        end
        V .= V .+ normalise_logweights!(wnorm);
        ps.filtered_index[] += 1;
    end
    V .= V .+ log.(wnorm);
    nothing
end

##
@with_kw struct Par{nx,nu,ny}
    A::MMatrix{nx,nx} = MMatrix{nx,nx}(Diagonal(rand(nx)))
    B::MMatrix{nx,nu} = MMatrix{nx,nu}(rand(nx * nu))
    C::MMatrix{ny,nx} = MMatrix{ny,nx}(rand(ny * nx))
    D::MMatrix{ny,nu} = MMatrix{ny,nu}(rand(ny * nu))
    Q::MMatrix{nx,nx} = MMatrix{nx,nx}(Diagonal(rand(nx)))
    R::MMatrix{ny,ny} = MMatrix{ny,ny}(Diagonal(rand(ny)))
    μ0::MVector{nx} = MVector{nx}(randn(nx))
    Σ0::MMatrix{nx,nx} = MMatrix{nx,nx}(Diagonal(rand(nx)))
end

nx = 3
nu = 2
ny = 3
T = 2000
M = 2000

LGSSM = build_LGSSM(nx)
θ = Par{nx,nu,ny}()

data = Data([randn(nu) for t in 1:T])
y = [zeros(Float64, ny) for t in 1:T]
x = [FloatParticle{nx}() for i in 1:T]

p = FloatParticle{nx}()
pnext = FloatParticle{nx}()

LGSSM.initial!(p, data, θ)
LGSSM.initial_logpdf(p, data, θ)
LGSSM.transition!(pnext, p, 1, data, θ)
LGSSM.transition_logpdf(pnext, p, 1, data, θ)
LGSSM.observe!(y[1], p, 1, data, θ)
LGSSM.observe_logpdf(y[1], p, 1, data, θ)

simulate!(y, LGSSM, data, θ)
simulate!(y, x, LGSSM, data, θ)

storage = ParticleStorage(FloatParticle{nx}, M, T)
ssm = SSMInstance(LGSSM, storage, data, T)

bpf!(ssm, θ)


##
results = DiffResults.JacobianResult(pcurr.x)

ForwardDiff.jacobian!(results, pcurr -> f(pcurr, zeros(nx), 1, data, θ), pcurr)

##
ForwardDiff.jacobian((xnext, xcurr) -> f!(xnext, xcurr, zeros(nx), t, data, θ), pnext.x, pcurr.x)
ForwardDiff.jacobian(xcurr -> predict(xcurr, zeros(nx), 1, data, θ), pcurr)

function state_jacobian(model::FunctionalSSM, t, data, θ)
    model.f()
    ForwardDiff.jacobian()
    predict!(pnext, pcurr, t, data, θ)

end
function ForwardDiff.jacobian(f::Any, p::Particle)
    ForwardDiff.jacobian(f, SVector(p))
end

##
function set!(p::FloatParticle{N}, x) where N
    p.x = x
end

##
particles = [FloatParticle{nx}() for t in 1:T]
particles[1].x .= 100 * rand(nx)
for i in 2:length(particles)
    predict!(particles[i], particles[i - 1], t, data, θ)
end

