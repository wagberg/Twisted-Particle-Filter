struct RtsSmoother{ST<:GaussianSmootherStorage,FT<:AbstractGaussianFilter} <: AbstractGaussianSmoother
    storage::ST
    filt::FT
end
function RtsSmoother(model::AbstractSSM{<:FloatParticle{N,S}}, data) where {N,S}
    storage = GaussianSmootherStorage(model, data)
    filt = KalmanFilter(storage)
    return RtsSmoother(storage, filt)
end
_filter(s::RtsSmoother) = s.filt
_storage(s::RtsSmoother) = s.storage
_model(s::RtsSmoother) = _model(_storage(s))
_data(s::RtsSmoother) = _data(_storage(s))

function smooth!(sdₜ, sdₜ₊₁, pdₜ₊₁, fdₜ, ::RtsSmoother, model, t, data, θ)
    A = transition_state_jacobian(SVector(fdₜ.μ), model, t, data, θ)
    U = UpperTriangular(pdₜ₊₁.Σ.chol.factors)
    G = (SMatrix(fdₜ.Σ.mat)*A' / U) / transpose(U)
    sdₜ.μ .= fdₜ.μ + G*(SVector(sdₜ₊₁.μ) - SVector(pdₜ₊₁.μ))
    P = PDMat(symmetrize(SMatrix(fdₜ.Σ.mat) + G*(SMatrix(sdₜ₊₁.Σ.mat) - SMatrix(pdₜ₊₁.Σ.mat))*G'))
    copy!(sdₜ.Σ, P)
end

# struct RTSSmoother{TK,TS} <: AbstractSmoother
#     storage::ST
#     kf::TK
#     dsmoothing::TS
# end
# function RTSSmoother(model::AbstractSSM{<:FloatParticle{N,S}}, data) where {N,S}
#     T = length(data.y)
#     dsmoothing = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T]
#     kf = KalmanFilter(model, data)
#     return RTSSmoother(kf, dsmoothing)
# end
# kf(f::RTSSmoother) = f.kf
# smoothing_density(f::RTSSmoother) = f.dsmoothing
# smoothing_density(f::RTSSmoother, t) = smoothing_density(f)[t]

# # Run RTS-smoother
# function (rts::RTSSmoother)(θ)
#     run_filter!(kf(rts), θ)
#     df = filter_density(rts)
#     dp = predictive_density(rts)
#     ds = smoothing_density(rts)

#     data = get_data(rts)
#     model = get_model(rts)
#     T = length(ds)

#     ds[T].μ .= df[T].μ
#     copy!(ds[T].Σ, df[T].Σ)
#     @inbounds for t = T-1:-1:1
#         A = transition_state_jacobian(SVector(df[t].μ), model, t, data, θ)
#         U = UpperTriangular(predictive_density(rts, t+1).Σ.chol.factors)
#         G = (SMatrix(filter_density(rts, t).Σ.mat)*A' / U) / transpose(U)
#         ds[t].μ .= df[t].μ + G*(SVector(ds[t+1].μ) - SVector(dp[t+1].μ))
#         P = PDMat(symmetrize(SMatrix(df[t].Σ.mat) + G*(SMatrix(ds[t+1].Σ.mat) - SMatrix(dp[t+1].Σ.mat))*G'))
#         copy!(ds[t].Σ, P)
#     end
#     return nothing
# end
