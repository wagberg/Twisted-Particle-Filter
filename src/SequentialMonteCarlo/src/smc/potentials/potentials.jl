abstract type AbstractPotential end

"""
Compute the log potential for the state pcurr
"""
log_potential(::AbstractPotential, xₜ::Particle, t, data, θ)

include("identity-potential.jl")
include("rts-potential.jl")
