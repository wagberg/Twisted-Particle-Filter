abstract type AbstractProposal end

"""
Simulate from proposal and return log probability density of sampled state.
"""
function simulate_proposal!(::AbstractProposal, x₁, data, θ) end
function simulate_proposal!(::AbstractProposal, xₜ₊₁, xₜ, t, data, θ) end

"""
Log pdf of proposal from time t+1 given t.
"""
function log_proposal_density(::AbstractProposal, xₜ₊₁, xₜ, t, data, θ) end
function log_proposal_density(::AbstractProposal, x₁, data, θ) end

"""
Initiate proposal by eg. precomputations
"""
function init!(::AbstractProposal) end

include("bootstrap.jl")
include("rts_proposal.jl")
include("locally_optimal.jl")
