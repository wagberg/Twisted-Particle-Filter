(smoother::AbstractGaussianSmoother)(θ) = (run_filter!(_storage(smoother), _filter(smoother), θ); run_smoother!(_storage(smoother), smoother, θ))

"""
Update smoohting density at time `t` given the smoothing and predictive density at time
`t+1` and the filter density at time `t`.
Arguments:
 * sdₜ: smothing density at current time
 * sdₜ₊₁: next time-step smoothing density
 * pdₜ₊₁: next time-step predictive desnity
 * fdₜ: current time filter density
"""
function smooth!(sdₜ, sdₜ₊₁, pdₜ₊₁, fdₜ, f::AbstractGaussianSmoother, model, t, data, θ) end
smooth!(s::AbstractGaussianStorage{Smoother}, f::AbstractGaussianSmoother, t, θ) =
    smooth!(smoothing_density(s, t), smoothing_density(s, t+1), predictive_density(s, t+1),
        filter_density(s, t), f, _model(s), t, _data(s), θ)

function run_smoother!(s::AbstractGaussianStorage{Smoother}, f::AbstractGaussianSmoother, θ)
    m = _model(s)
    d = _data(s)
    T = length(smoothing_density(s))
    smoothing_density(s, T).μ .= filter_density(s, T).μ
    copy!(smoothing_density(s, T).Σ, filter_density(s, T).Σ)

    for t ∈ T-1:-1:1
        smooth!(s, f, t, θ)
    end
end

include("rts_smoother.jl")
