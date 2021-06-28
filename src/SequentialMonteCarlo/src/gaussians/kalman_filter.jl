struct KalmanFilter{ST<:AbstractGaussianStorage} <: AbstractGaussianFilter
    storage::ST
end
function KalmanFilter(model, data)
    storage = GaussianFilterStorage(model, data)
    return KalmanFilter(storage)
end
# kf(f::KalmanFilter) = f
_storage(f::KalmanFilter) = f.storage

# (kf::KalmanFilter)(θ) = run_filter!(storage(kf), kf, θ)

@inline function predict(df, ::KalmanFilter, model, t, data, θ)
    # m = SVector(df.μ)
    # S = df.Σ
    m, S = params(df)
    m = SVector(m)
    Q = PDMat(transition_noise_covariance(m, model, t, data, θ))
    A = transition_state_jacobian(m, model, t, data, θ)
    M = transition_noise_jacobian(m, model, t, data, θ)
    P = PDMat(symmetrize(X_A_Xt(S, A) + X_A_Xt(Q, M)))
    μ = transition_function(m, model, t, data, θ)
    return MvNormal(μ, P)
end

function observe(pd, ::KalmanFilter, model, t, data, θ)
    μp = SVector(pd.μ)
    Σp = pd.Σ
    yp = observation_function(μp, model, t, data, θ)
    R = PDMat(observation_noise_covariance(μp, model, t, data, θ))
    C = observation_state_jacobian(μp, model, t, data, θ)
    N = observation_noise_jacobian(μp, model, t, data, θ)
    S = PDMat(symmetrize(X_A_Xt(Σp, C) + X_A_Xt(R, N)))

    ll::Float64 = logpdf(MvNormal(yp,S), data.y[t])
    K = transpose(UpperTriangular(S.chol.factors) \ (transpose(UpperTriangular(S.chol.factors)) \ (C*Σp.mat)))
    μ = μp + K*(data.y[t] - yp)
    # Σf = (I - K*C)*Σp*(I - K*C)' + K*R*K' # Joseph stabilized update
    # filter_Sigma[t] .= predic_Sigma[t] - K*S*transpose(K) # Särkkä
    P = Σp - X_A_Xt(S,K)
    return (ll, MvNormal(μ, P))
end
