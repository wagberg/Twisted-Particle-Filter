struct CopyResampler <: Resampler end

function resample!(::CopyResampler, Aₜ, Wₜ, logWₜ, logWₜ₊₁)
    Aₜ .= 1:length(Aₜ)
    logWₜ₊₁ .= logWₜ
    return nothing
end

struct ConditionalCopyResampler <: ConditionalResampler end
conditional(r::CopyResampler) = ConditionalCopyResampler()

resample!(::ConditionalCopyResampler, Aₜ, Wₜ, logWₜ, logWₜ₊₁) = resample(CopyResampler(), Aₜ, Wₜ, logWₜ, logWₜ₊₁)
