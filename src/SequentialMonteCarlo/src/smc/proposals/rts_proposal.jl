struct RTSProposal{MT, RT, DT, HT} <: AbstractProposal
    model::MT
    rts::RT
    d::DT
    H::HT
end
function RTSProposal(rts)
    model = _model(rts)
    ds = smoothing_density(rts)
    T = length(ds)
    N, S = length(ds[1]), eltype(ds[1])
    d = [MvNormal(MVector{N,S}(undef), one(MMatrix{N,N,S})) for _ in 1:T]
    H = [MMatrix{N,N,S}(undef) for _ in 1:T]
    return RTSProposal(model, rts, d, H)
end
RTSProposal(model, data) = RTSProposal(RTSSmoother(model, data))
Base.show(io::IO, ::RTSProposal) = print(io, "Optimal proposal from RTS-smoother")

function init!(p::RTSProposal, θ)
    rts = p.rts
    data = _data(rts)
    H = p.H
    d = p.d
    model = p.model

    T = length(smoothing_density(rts))
    for t = T-1:-1:1
        A = transition_state_jacobian(filter_density(rts, t).μ, model, t, data, θ)
        # Gₜ = Σ(t|t)*A(t)ᵀ / Σ(t+1|t)
        U = UpperTriangular(predictive_density(rts, t+1).Σ.chol.factors)
        G = (SMatrix(filter_density(rts, t).Σ.mat)*A' / U) / U'

        # H(t) = Σ(t+1|T) G(t)ᵀ Σ(t|T)⁻¹
        U = UpperTriangular(smoothing_density(rts,t).Σ.chol.factors)
        H̃ = SMatrix(smoothing_density(rts,t+1).Σ.mat)*G' / U
        H[t] .= H̃ / transpose(U)
        d[t].μ .= smoothing_density(rts, t+1).μ - H[t]*smoothing_density(rts, t).μ
        Σ = PDMat(SMatrix(smoothing_density(rts, t+1).Σ.mat) - H̃*transpose(H̃))
        copy!(d[t].Σ, Σ)
    end
end

log_proposal_density(p::RTSProposal, x₁, data, θ) = logpdf(smoothing_density(p.rts, 1), x₁.x)
log_proposal_density(p::RTSProposal, xₜ₊₁, xₜ, t, data, θ) = logpdf(p.d[t], xₜ₊₁.x - p.H[t]*xₜ.x)

function simulate_proposal!(p::RTSProposal, xₜ₊₁, xₜ, t, data, θ)
    d = p.d[t]
    H = p.H[t]
    rand!(d, xₜ₊₁.x)
    ℓ = logpdf(d, xₜ₊₁.x)
    xₜ₊₁.x .+= H*xₜ.x
    return ℓ
end

function simulate_proposal!(p::RTSProposal, x₁, data, θ)
    d = smoothing_density(p.rts, 1)
    rand!(d, x₁.x)
    return logpdf(d, x₁.x)
end
