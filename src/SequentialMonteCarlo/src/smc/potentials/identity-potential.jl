struct IdentityPotential <: AbstractPotential end

log_potential(::IdentityPotential, xₜ, t, data, θ) = 0.0

Base.show(io::IO, p::IdentityPotential) = print(io, "Identity potential")
