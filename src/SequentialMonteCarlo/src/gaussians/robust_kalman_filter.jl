# struct RobustKalmanFilter{DT,MT<:AbstractFunctionalSSM, FT, PT, LT } <: AbstractKalmanFilter
#     data::DT
#     model::MT
#     dfilter::FT
#     dprediction::PT
#     ll::LT
# end
# function RobustKalmanFilter(model::AbstractFunctionalSSM{<:FloatParticle{N,S}}, data) where {N,S}
#     T = length(data.y)
#     dfilter = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T]
#     dprediction = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T+1]
#     ll = Vector{S}(undef, T)
#     return RobustKalmanFilter(data, model, dfilter, dprediction, ll)
# end
# kf(f::RobustKalmanFilter) = f


# function predict!(kf::RobustKalmanFilter, t, θ)
#     model = get_model(kf)
#     data = get_data(kf)
#     dp = predictive_density(kf, t+1)
#     df = filter_density(kf, t)
#     dp.μ .= transition_function(SVector(df.μ), model, t, data, θ)

#     Q = PDMat(transition_noise_covariance(df.μ, model, t, data, θ))
#     A = transition_state_jacobian(df.μ, model, t, data, θ)
#     M = transition_noise_jacobian(df.μ, model, t, data, θ)
#     P = X_A_Xt(df.Σ, A) + X_A_Xt(Q, M)
#     copy!(dp.Σ, PDMat(symmetrize(P)))
# end

# function observe!(kf::RobustKalmanFilter, t, θ)
#     println("hej")
#     model = get_model(kf)
#     data = get_data(kf)
#     dp = predictive_density(kf, t)
#     df = filter_density(kf, t)

#     v = data.y[t] - observation_function(dp.μ, model, t, data, θ)

#     R = observation_noise_covariance(dp.μ, model, t, data, θ)
#     C = observation_state_jacobian(dp.μ, model, t, data, θ)
#     N = observation_noise_jacobian(dp.μ, model, t, data, θ)

#     S = PDMat(chol_from_qr(qr([C*UpperTriangular(dp.Σ.chol.factors)' N*R]')))
#     ll = logpdf(MvNormal(S), v)
#     K = transpose(UpperTriangular(S.chol.factors) \ (transpose(UpperTriangular(S.chol.factors)) \ (C*dp.Σ.mat)))

#     df.μ .= dp.μ + K*v
#     copy!(df.Σ, dp.Σ)
#     KS = K*transpose(UpperTriangular(S.chol.factors))
#     C = df.Σ.chol
#     choldowndate!(C, KS)
#     copy!(df.Σ, PDMat(C))
# end
