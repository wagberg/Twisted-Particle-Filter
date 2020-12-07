__precompile__(true)
module SequentialMonteCarlo

using LinearAlgebra
using Distributions
using Random
using StaticArrays
using Parameters
using ForwardDiff
using StatsFuns

const AVec = AbstractVector
const AMat = AbstractMatrix
const AFloat = AbstractFloat

abstract type Particle end
abstract type SSMParameter end

include("utils.jl")
include("statespace.jl")
include("floatparticle.jl")
include("kalman.jl")
include("resampling.jl")
include("particle_filter.jl")
include("linear_gaussian.jl")

export
    SSM,
    FloatParticle,
    LGSSM,
    # SSMParameter,
    LGSSMParameter,
    KalmanStorage,
    ParticleStorage,
    simulate!,
    ekf!,
    smooth!,
    bpf!,
    tpf!

end # module
