struct RobustKalmanFilter{ST<:AbstractGaussianStorage} <: AbstractGaussianFilter
    storage::ST
end
function RobustKalmanFilter(model, data)
    storage = GaussianFilterStorage(model, data)
    return RobustKalmanFilter(storage)
end
_storage(f::RobustKalmanFilter) = f.storage

function predict(df, ::RobustKalmanFilter, model, t, data, θ)
    m, S = params(df)
    m = SVector(m)
    U = SMatrix(S.chol.factors)

    A = transition_state_jacobian(m, model, t, data, θ)
    M = transition_noise_jacobian(m, model, t, data, θ)
    Q = transition_noise_covariance(m, model, t, data, θ) |> PDMat

    C = vcat(U*A', Q.chol.factors*M')
    q = qr(C)
    U´ = chol_from_qr(q)
    P = PDMat(transpose(U´.factors)*U´.factors, U´)
    !isposdef(P.mat) && @error "P=$P not pos def. U´=$U´, C=$C, U=$U, A=$A, t=$t"
    μ = transition_function(m, model, t, data, θ)
    return MvNormal(μ, P)
end

function observe(dp, ::RobustKalmanFilter, model, t, data, θ)
    m, S = Distributions.params(dp)
    m = SVector(m)
    U = SMatrix(S.chol.factors)

    C = observation_state_jacobian(m, model, t, data, θ)
    N = observation_noise_jacobian(m, model, t, data, θ)
    R = PDMat(observation_noise_covariance(m, model, t, data, θ))

    ne = size(R,1)
    ny = size(C,1)
    nx = size(U,1)
    A = hcat(vcat(R.chol.factors*N', U*C'), vcat(zero(SMatrix{ne, nx}), U))
    q = qr(A) |> q->sign.(diag(q.R)).*q.R

    # Update mean
    W = q[StaticArrays.SUnitRange(1,ne),StaticArrays.SUnitRange(ny+1,ny+nx)]
    B = q[StaticArrays.SUnitRange(1,ne),StaticArrays.SUnitRange(1,ny)]
    K = transpose(B\W)
    yp = observation_function(m, model, t, data, θ)
    m´ = m + K*(data.y[t] - yp)

    # Compute likelihood
    S = q[StaticArrays.SUnitRange(1,ne),StaticArrays.SUnitRange(1,ny)] |> x->x'*x
    !isposdef(S) && @error "S=$S not pos def. A=$A, dp.Σ = $(dp.Σ)"
    ll = logpdf(MvNormal(yp,S), data.y[t])

    # Update covanriace
    v1 = StaticArrays.SUnitRange(ne+1, ne+nx)
    v2 = StaticArrays.SUnitRange(ny+1, ny+nx)
    P = q[v1,v2]
    C = Cholesky(P, 'U', 0)
    P´ = PDMat(P'*P, C)
    return (ll, MvNormal(m´, P´))
end


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
