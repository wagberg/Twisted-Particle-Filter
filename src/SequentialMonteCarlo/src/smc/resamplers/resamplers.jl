abstract type ResamplerType end
abstract type Unconditional <: ResamplerType end
abstract type Conditional <: ResamplerType end
"""
An abstract type representing a resampling algorithm.
"""
abstract type AbstractResampler{T<:ResamplerType} end

const Resampler = AbstractResampler{Unconditional}
const ConditionalResampler = AbstractResampler{Conditional}

conditional(s::ConditionalResampler) = s
"""
Sample indices with probability ` p(Aₜ = i) = Wₜⁱ ∀ i`.
Assumes `sum(w) = 1`.
Resamplers with euqal weights after resampling implements the three argument `resample!`,
while non uniform resampled weights, eg. copy and optimal, implements the five argument
`resample!`.
"""
function resample!(r::AbstractResampler, A, W) end
"""
Resample function for resample schemes with non-uniform resampled weights.
"""
function resample!(r::AbstractResampler, Aₜ, Wₜ, logWₜ, logWₜ₊₁)
    resample!(r, Aₜ, Wₜ)
    logWₜ₊₁ .= -log(length(logWₜ))
end

include("multinomial.jl")
include("systematic.jl")
include("copy.jl")
include("ess_threshold.jl")
