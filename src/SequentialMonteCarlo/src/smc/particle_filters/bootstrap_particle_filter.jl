# struct BootstrapParticleFilter <: AbstractParticleFilter end

# struct BootstrapParticleFilter{PT} <: AbstractParticleFilter
#     pf::PT
# end
function BootstrapParticleFilter(model, n_particles, data, resampler=ResampleWithESSThreshold())
    proposal = BootstrapProposal(model)
    potential = IdentityPotential()
    pf = ParticleFilter(model, n_particles, data, proposal, potential, resampler)
    return pf
end

# _pf(f::BootstrapParticleFilter) = f.pf
# name(::BootstrapParticleFilter) = "Bootstrap particle filter"
