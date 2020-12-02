# Generic update function

"""
    update(filter::AbstractFilter, b0::GaussianBelief, u::AbstractVector,
        y::AbstractVector)

Uses AbstractFilter filter to update gaussian belief b0, given control vector
u and measurement vector y.
"""
function update(filter::AbstractFilter, bp::GaussianBelief,
                u::AbstractVector{<:Number}, y::AbstractVector{<:Number})
bf = measure(filter, bp, y, u)
bp = predict(filter, bf, u)

return bp
end

# Kalman filter functions
"""
    predict(filter::KalmanFilter, b0::GaussianBelief, u::AbstractVector)

Uses Kalman filter to run prediction step on gaussian belief bf, given control
vector u.
"""
function predict(filter::KalmanFilter, bf::GaussianBelief;
            u::AbstractVector{<:Number} = [false])
μp = predict(filter.d, bf.μ; u)
Σp = filter.d.A * bf.Σ * filter.d.A' + filter.d.M * cov(filter.d.d) * filter.d.M'
return GaussianBelief(μp, Σp)
end

"""
    measure(filter::KalmanFilter, bp::GaussianBelief, y::AbstractVector;
        u::AbstractVector = [false])

Uses Kalman filter to run measurement update on predicted gaussian belief bp,
given measurement vector y. If u is specified and filter.o.D has been declared,
then matrix D will be factored into the y predictions
"""
function measure(filter::KalmanFilter, bp::GaussianBelief, y::AbstractVector{<:Number};
                u::AbstractVector{<:Number} = [false])

m = filter.o
# Predict measurement
yp = measure(m, bp.μ; u)

# Likelihood
S = m.C*bp.Σ*m.C' + m.N*cov(m.d)*m.N'
ll = logpdf(MvNormal(yp, S), y)

# Kalman Gain
K = (S\(m.C * bp.Σ))'

# Measurement update
μf = bp.μ + K*(y - yp)
Σf = (I - K*m.C)*bp.Σ
return GaussianBelief(μf, Σf), ll
end

function smooth(smoother::RtsSmoother, bs::GaussianBelief,
    bf::GaussianBelief; u::AbstractVector{<:Number} = [false])
bp = predict(smoother.f, bf; u)
G = (bp.Σ \ (smoother.f.d.A * bf.Σ))'
μ = bf.μ + G*(bs.μ - bp.μ)
Σ = bf.Σ + G*(bs.Σ - bp.Σ)*G'
return GaussianBelief(μ, Σ)
end