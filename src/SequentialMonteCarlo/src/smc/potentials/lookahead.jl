struct LookAhead{N,MT,FT} <: AbstractPotential
    model::MT
    f::FT
end

function LookAhead(N, model, f::Type{<:AbstractGaussianFilter} = KalmanFilter)
    filter = f(EmptyGaussianStorage())
    return LookAhead{N,typeof(model), typeof(filter)}(model, filter)
end
Base.show(io::IO, p::LookAhead{N}) where N = print(io, "$(N) step look-ahead potential")


function log_potential(p::LookAhead{N}, xₜ::FloatParticle{dx,T}, t, data, θ) where {N,dx,T}
    x = toSVector(xₜ)
    C = Cholesky(zero(SMatrix{dx,dx,T}), 'U', 0)
    Σ = PDMat(zero(SMatrix{dx,dx,T}), C)
    d = MvNormal(x, Σ)
    ll = 0
    for s in t+1:min(t+N,length(data.y))
        d = predict(d, p.f, p.model, s-1, data, θ)
        l,d = observe(d, p.f, p.model, s, data, θ)
        ll += l
    end
    return ll
end
