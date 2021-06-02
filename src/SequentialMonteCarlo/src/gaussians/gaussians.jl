abstract type AbstractGaussian{T<:InferenceType} end
const AbstractGaussianFilter = AbstractGaussian{Filter}
const AbstractGaussianSmoother = AbstractGaussian{Smoother}

abstract type AbstractGaussianStorage{T<:InferenceType} end

_model(s::AbstractGaussianStorage) = s.model
_data(s::AbstractGaussianStorage) = s.data

struct EmptyGaussianStorage <: AbstractGaussianStorage{Filter} end

struct GaussianFilterStorage{MT, DT, FT, PT, T} <: AbstractGaussianStorage{Filter}
    model::MT
    data::DT
    filter_densities::FT
    prediction_densities::PT
    ll::Vector{T}
end
function GaussianFilterStorage(model::AbstractSSM{<:FloatParticle{N,S}}, data) where {N,S}
    T = length(data.y)
    filter_densities = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T]
    prediction_densities = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T+1]
    ll = Vector{S}(undef, T)
    return GaussianFilterStorage(model, data, filter_densities, prediction_densities, ll)
end

struct GaussianSmootherStorage{MT, DT, FT, PT, ST, T} <: AbstractGaussianStorage{Smoother}
    model::MT
    data::DT
    filter_densities::FT
    prediction_densities::PT
    smoothing_densities::ST
    ll::Vector{T}
end
function GaussianSmootherStorage(model::AbstractSSM{<:FloatParticle{N, S}}, data) where {N,S}
    T = length(data.y)
    filter_densities = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T]
    prediction_densities = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T+1]
    smoothing_densities = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T]
    ll = Vector{S}(undef, T)
    return GaussianSmootherStorage(model, data, filter_densities, prediction_densities, smoothing_densities, ll)
end

"""
Returns the filter density at time `t`.
"""
filter_density(s::AbstractGaussianStorage) = s.filter_densities
"""
Filter density ad time `t`.
"""
filter_density(s::AbstractGaussianStorage, t::Int) = filter_density(s)[t]
filter_density(f::AbstractGaussian, args...) = filter_density(_storage(f), args...)
"""
Returns the predictive density
"""
predictive_density(s::AbstractGaussianStorage) = s.prediction_densities
"""
Predictive density ad time `t`.
"""
predictive_density(s::AbstractGaussianStorage, t::Int) = predictive_density(s)[t]
predictive_density(f::AbstractGaussian, args...) = predictive_density(_storage(f), args...)

"""
Returns the log likelihood
"""
log_likelihood(s::AbstractGaussianStorage) = s.ll
log_likelihood(s::AbstractGaussianStorage, t::Int) = log_likelihood(s)[t]
log_likelihood(f::AbstractGaussian, args...) = log_likelihood(_storage(f),args...)

set_log_likelihood!(s::AbstractGaussianStorage, t::Int, ll) = (log_likelihood(s)[t] = ll)

"""
Returns the smoothing density
"""
smoothing_density(s::AbstractGaussianStorage{Smoother}) = s.smoothing_densities
"""
Smoohting density ad time `t`.
"""
smoothing_density(s::AbstractGaussianStorage{Smoother}, t::Int) = smoothing_density(s)[t]
smoothing_density(f::AbstractGaussianSmoother, args...) = smoothing_density(_storage(f), args...)


include("filtering.jl")
include("smoothing.jl")
