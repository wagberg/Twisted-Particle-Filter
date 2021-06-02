struct ResampleWithESSThreshold{S<:ResamplerType, RT<:AbstractResampler{S}, T<:Real} <: AbstractResampler{S}
    resampler::RT
    threshold::T
end
ResampleWithESSThreshold(resampler::AbstractResampler = SystematicResampler()) = ResampleWithESSThreshold(resampler, 0.5)

function conditional(r::ResampleWithESSThreshold)
    if r.resampler isa ConditionalResampler
        return r
    else
        ResampleWithESSThreshold(conditional(r.resampler), r.threshold)
    end
end
"""
Compute the effective sample size.
The weghts `w` are assumed normalized.
"""
function ess(w)
    return 1/mapreduce(x->x^2, +, w)
end

function resample!(r::ResampleWithESSThreshold, Aₜ, Wₜ, logWₜ, logWₜ₊₁)
    if ess(Wₜ) < length(Wₜ)*r.threshold
        resample!(r.resampler, Aₜ, Wₜ,logWₜ, logWₜ₊₁)
    else
        resample!(CopyResampler(), Aₜ, Wₜ, logWₜ, logWₜ₊₁)
    end
    return nothing
end
