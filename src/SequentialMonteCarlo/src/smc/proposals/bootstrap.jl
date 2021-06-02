struct BootstrapProposal{MT} <: AbstractProposal
    model::MT
end


function log_proposal_density(bp::BootstrapProposal, x₁::Particle, data, θ)
    # @error "`log_proposal_density` should never be called"
    # It is only called when running conditional particle filters
    # log_initial_density(x₁, bp.model, data, θ)
   return 0.0

end
# The transition density should never be called!
function log_proposal_density(bp::BootstrapProposal, xₜ₊₁, xₜ, t, data, θ)
    # @error "log_proposal_density should never be called when using BootstrapProposal"
    # It is only called when running conditional particle filters
    # log_transition_density(xₜ₊₁, xₜ, bp.model, t, data, θ)
    return 0.0
end

function simulate_proposal!(bp::BootstrapProposal, xₜ₊₁, xₜ, t, data, θ)
    simulate_transition!(xₜ₊₁, xₜ, bp.model, t, data, θ)
    return 0.0 # Return NaN since log_proposal_density shhould never be called
end

function simulate_proposal!(bp::BootstrapProposal, x₁, data, θ)
    simulate_initial!(x₁, bp.model, data, θ)
    return 0.0# log_initial_density(x₁, bp.model, data, θ)
end

Base.show(io::IO, p::BootstrapProposal) = print(io, "Bootstrap proposal")
