(filter::AbstractGaussianFilter)(θ) = run_filter!(_storage(filter), filter, θ)

"""
`predict(fd, f, model, t, data, θ) -> pd`
Update predicitve density at time `t+1` given filter density at time `t`.
Arguments:
 * `pd`: predictive density to update, eg. at time t+1
 * `fd`: filter density, eg at time t
 * `f`: filter
 * `t`: time index
 * `data`: data needed by model
 * `θ`: parameter
"""
function predict(fd, f::AbstractGaussianFilter, model, t, data, θ) end

predict!(dp, df, f::AbstractGaussianFilter, model, t, data, θ) = copy!(dp, predict(df, f, model, t, data, θ))
predict!(s::AbstractGaussianStorage, f::AbstractGaussianFilter, t, θ) =
    predict!(predictive_density(s, t+1), filter_density(s, t), f, _model(s), t, _data(s), θ)


"""
`observe(pd, f::AbstractGaussianFilter, model, t, data, θ) -> fd`

Update filter density at time `t` given observation and predictive density at time `t`.
Arguments:
 * fd: filter density
 * pd: predictive density
 * f: filter
 * t: time index
 * data: data needed by model
 * θ: parameter
"""
function observe(pd, f::AbstractGaussianFilter, model, t, data, θ) end
function observe!(fd, pd, f::AbstractGaussianFilter, model, t, data, θ)
    ll, d = observe(pd, f, model, t, data, θ)
    copy!(fd, d)
    return ll
end
observe!(s::AbstractGaussianStorage, f::AbstractGaussianFilter, t, θ) = observe!(filter_density(s, t), predictive_density(s, t), f, _model(s), t, _data(s), θ)

function run_filter!(s::AbstractGaussianStorage, f::AbstractGaussianFilter, θ)
    m = _model(s)
    d = _data(s)
    T = length(d.y)

    predictive_density(s, 1).μ .= initial_mean(m, d, θ)
    copy!(predictive_density(s, 1).Σ, PDMat(initial_covariance(m, d, θ)))
    ll = 0.0::Float64

    for t in eachindex(d.y)
        ll += observe!(s, f, t, θ)
        set_log_likelihood!(s, t, ll)
        predict!(s, f, t, θ)
    end
end

include("kalman_filter.jl")
include("robust_kalman_filter.jl")
