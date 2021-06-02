__precompile__(true)
module SequentialMonteCarlo

using LinearAlgebra
using Distributions
using Random
using DataFrames
using StaticArrays
using StaticArrays: StaticVecOrMat, StaticVecOrMatLike
using TimerOutputs
using Parameters
using ForwardDiff
using StatsFuns
using DiffResults
using PDMats
using LoopVectorization
import Base: copy!, (==)

export Particle
export FloatParticle
export toSVector
export AbstractSSM
# export KalmanStorage
export KalmanFilter, RtsSmoother
export filter_density, predictive_density, smoothing_density ,log_likelihood
export ParticleFilter, BootstrapParticleFilter, ConditionalParticleFilter, ConditionalParticleFilterWithAncestorSampling
export AbstractProposal, BootstrapProposal, RTSProposal, LocallyOptimalProposal
export AbstractPotetial, IdentityPotential, RTSPotential
export SystematicResampler, MultinomialResampler, ResampleWithESSThreshold
export ConditionalMultinomialResampler, ConditionalSystematicResampler
export FullParticleStorage, SlimParticleStorage
export LinearGaussian, LinearGaussianPar
export run_filter!, name, states, weights, reference
export init!
export simulate
export statenames
export default_parameters

# const AVec = AbstractVector
# const AMat = AbstractMatrix
# const AFloat = AbstractFloat

# """
# An abstract type for an MVector or SVector with length N and element type T.
# """
# const VecND{N,T} = StaticArray{Tuple{N},T,1};

# """
# An abstract type for an N * N MMatrix or SMatrix with element type T.
# """
# const MatND{T, N} = StaticArray{Tuple{N, N}, T, 2};

# abstract type AbstractFilter end
# abstract type AbstractSmoother end

abstract type InferenceType end
abstract type Filter <: InferenceType end
abstract type Smoother <: InferenceType end

include("utils.jl")
include("models/models.jl")
include("gaussians/gaussians.jl")
include("smc/smc.jl")

end # module
