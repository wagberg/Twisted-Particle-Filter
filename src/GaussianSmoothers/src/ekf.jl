# Extended Kalman filter functions

"""
    predict(filter::ExtendedKalmanFilter, b0::GaussianBelief, u::AbstractVector)

Uses Extended Kalman filter to run prediction step on gaussian belief b0,
given control vector u.
"""
function predict(filter::ExtendedKalmanFilter, b0::GaussianBelief,
                u::AbstractVector{<:Number})

    m = filter.d
    # Motion update
    μp = predict(m, b0.μ, u)
    F = jacobian(StateJacobian(), m, b0.μ, u)
    M = jacobian(NoiseJacobian(), m, b0.μ, u)

    Σp = F * b0.Σ * F' + M * cov(m.d) * M'
    return GaussianBelief(μp, Σp)
end

"""
    predict(filter::ExtendedKalmanFilter, b0::GaussianBelief, u::AbstractVector)

Uses Extended Kalman filter to run prediction step on gaussian belief b0,
given control vector u.
"""
function predict!(b0::GaussianBelief, filter::ExtendedKalmanFilter, u::AbstractVector{<:Number})

    m = filter.d
    # Motion update
    μp = predict(m, b0.μ, u)
    F = jacobian(StateJacobian(), m, b0.μ, u)
    M = jacobian(NoiseJacobian(), m, b0.μ, u)

    Σp = F * b0.Σ * F' + M * cov(m.d) * M'
    return GaussianBelief(μp, Σp)
end

"""
    measure(filter::ExtendedKalmanFilter, bp::GaussianBelief, y::AbstractVector;
        u::AbstractVector = [false])

Uses Extended Kalman filter to run measurement update on predicted gaussian
belief bp, given measurement vector y. If u is specified and filter.o.D has
been declared, then matrix D will be factored into the y predictions.
"""
function measure(filter::ExtendedKalmanFilter, bp::GaussianBelief, y::AbstractVector{a},
                u::AbstractVector{b} = [false]) where {a<:Number, b<:Number}

    m = filter.o    
    # Measurement update
    yp = measure(m, bp.μ, u)
    H = jacobian(StateJacobian(), m, bp.μ, u)
    N = jacobian(NoiseJacobian(), m, bp.μ, u)

    # Data distribution
    S = H * bp.Σ * H' + N * cov(m.d) * N'
    ll = logpdf(MvNormal(yp, S), y)
    
    # Kalman Gain
    # K = bp.Σ * H' * inv(H * bp.Σ * H' + filter.o.V)
    K = (S\(H * bp.Σ))'

    # Measurement update
    μn = bp.μ + K * (y - yp)
    Σn = (I - K * H) * bp.Σ
    return GaussianBelief(μn, Σn), ll
end

function smooth(smoother::ExtendedRtsSmoother, bs::GaussianBelief,
    bf::GaussianBelief, u::AbstractVector{<:Number} = [false])
bp = predict(smoother.f, bf, u)
F = jacobian(StateJacobian(), smoother.f.d, bg.μ, u)
G = (bp.Σ \ (bf.Σ * F))'
μ = bf.μ + G*(bs.μ - bp.μ)
Σ = bf.Σ + G*(bs.Σ - bp.Σ)*G'
return GaussianBelief(μ, Σ)
end