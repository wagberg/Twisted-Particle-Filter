"""
Preallocate storage for extended Kalman filter.

* `filter_mean`, `filter_Sigma`: Contains p(x_{t} | y_{1:t})
* `predic_mean`, `predic_Sigma`: Contains p(x_{t} | y_{1:t-1})
* `smooth_mean`, `smooth_Sigma`: Contains p(x_{t} | y_{1:T})
* `log_likelihood`: Contains p(y_{1:t})
"""
struct KalmanStorage{T}
    filter_mean::Vector{Vector{T}}
    predic_mean::Vector{Vector{T}}
    smooth_mean::Vector{Vector{T}}
    filter_Sigma::Vector{Symmetric{T, Matrix{T}}}
    predic_Sigma::Vector{Symmetric{T, Matrix{T}}}
    smooth_Sigma::Vector{Symmetric{T, Matrix{T}}}
    log_likelihood::Vector{T}

    function KalmanStorage(::Type{FloatParticle{N}}, t::Integer) where N
        filter_mean = [zeros(Float64, N) for i in 1:t]
        predic_mean = [zeros(Float64, N) for i in 1:t+1]
        smooth_mean = [zeros(Float64, N) for i in 1:t]
        filter_Sigma = [Symmetric(zeros(Float64, N, N)) for i in 1:t]
        predic_Sigma = [Symmetric(zeros(Float64, N, N)) for i in 1:t+1]
        smooth_Sigma = [Symmetric(zeros(Float64, N, N)) for i in 1:t]
        log_likelihood = zeros(Float64, t)
        new{Float64}(filter_mean, predic_mean, smooth_mean,
            filter_Sigma, predic_Sigma, smooth_Sigma, log_likelihood)
    end
end

"""
Extended Kalman filter.
* `storage`: Preallocated storage
* `model`: State-space model
* `data`: Data needed by the state-space model. Must contain a field y for measured output.
* `θ`: Parameters for the state-space model
* `T`: Run filter up until time T
"""
function ekf!(storage::KalmanStorage, model::SSM, data, θ, T = length(data.y))
    filter_mean = storage.filter_mean
    filter_Sigma = storage.filter_Sigma
    
    predic_mean = storage.predic_mean
    predic_Sigma = storage.predic_Sigma
    
    log_likelihood = storage.log_likelihood

    predic_mean[1] .= initial_mean(model, data, θ)
    predic_Sigma[1] .= Symmetric(initial_covariance(model, data, θ))

    ll = zero(eltype(log_likelihood))

    for t = 1:T
        # measurement update
        yp = observation_function(predic_mean[t], model, t, data, θ)
        R = observation_covariance(predic_mean[t], model, t, data, θ)
        C = observation_state_jacobian(predic_mean[t], model, t, data, θ)
        S = Symmetric(C*predic_Sigma[t]*C' .+ R)
        ll += logpdf(MvNormal(yp, S), data.y[t])
        log_likelihood[t] =  ll
        K = (S\(C*predic_Sigma[t]))'
        filter_mean[t] .= predic_mean[t] .+ K*(data.y[t]-yp)
        filter_Sigma[t] .= Symmetric((I - K*C)*predic_Sigma[t])
        
        # predict
        predic_mean[t+1] .= transition_function(filter_mean[t], model, t, data, θ)
        Q = transition_covariance(filter_mean[t], model, t, data, θ)
        A = transition_state_jacobian(filter_mean[t], model, t, data, θ)
        predic_Sigma[t+1] .= Symmetric(A*filter_Sigma[t]*A' + Q)
    end
end

"""
Run extended RTS-smoother.
Assumes an extended Kalman filter has already been run.
* `storage`: Preallocated storage with already computed filtering and predictive distirbuitons
* `model`: State-space model
* `data`: Data for the state space model. (Typically input and output)
"""
function smooth!(storage, model, data, θ)
    filter_mean = storage.filter_mean
    predic_mean = storage.predic_mean
    smooth_mean = storage.smooth_mean
    filter_Sigma = storage.filter_Sigma
    predic_Sigma = storage.predic_Sigma
    smooth_Sigma = storage.smooth_Sigma
    
    T = length(smooth_mean)
    smooth_mean[T] = filter_mean[T]
    smooth_Sigma[T] = filter_Sigma[T]
    for t = T-1:-1:1
        A = transition_state_jacobian(filter_mean[t], model, t, data, θ)
        G = (predic_Sigma[t+1] \ (A * filter_Sigma[t]))'
        smooth_mean[t] .= filter_mean[t] + G*(smooth_mean[t+1] - predic_mean[t+1])
        smooth_Sigma[t] .= Symmetric(filter_Sigma[t] + G*(smooth_Sigma[t+1] - predic_Sigma[t+1])*G')
    end
end