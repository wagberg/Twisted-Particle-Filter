struct RTSPotential{RS} <: AbstractPotential
    rts::RS
end
function RTSPotential(model, data)
    rts = RtsSmoother(model, data)
    return RTSPotential(rts)
end
Base.show(io::IO, p::RTSPotential) = print(io, "RTS potential")


function log_potential(p::RTSPotential, xₜ::FloatParticle, t, data, θ)
    ds = smoothing_density(p.rts, t)
    df = filter_density(p.rts, t)
    llT = log_likelihood(p.rts)[end]
    llt = log_likelihood(p.rts)[t]

    return logpdf(ds, xₜ.x) + llT - logpdf(df, xₜ.x) - llt
end
