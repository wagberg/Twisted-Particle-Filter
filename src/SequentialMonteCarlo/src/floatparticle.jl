struct FloatParticle{N}  <: Particle
    x::MVector{N, Float64}
end

function FloatParticle{N}() where {N}
    FloatParticle(zero(MVector{N, Float64}));
end

function copy!(dest::FloatParticle{N}, src::FloatParticle{N}) where N
    for i in eachindex(src.x)
        @inbounds dest.x[i] = src.x[i]
    end
end

function swap!(dest::FloatParticle{N}, src::FloatParticle{N}) where N
    for i in eachindex(src.x)
        @inbounds src.x[i], dest.x[i] = dest.x[i], src.x[i]
    end
end

# function copy!(dest::FloatParticle{N, T}, src::StaticArray{Tuple{N}, T, 1}) where {N, T}
#     for i in eachindex(dest.x)
#         @inbounds dest.x[i] = src[i];
#     end
#     dest;
# end

"""
Convert the particle to a static vector.
This is used to compute gradients of the transition function.
"""
function toSVector(p::FloatParticle{N}) where N
    SVector{N}(p.x);
end

function statenames(::Type{FloatParticle{N}}) where N
    Symbol.("x" .* string.(collect(1:N)))
end

function simulate_observation!(y::AVec{<:AFloat}, p::FloatParticle{N}, model::SSM, t::Integer, data, θ) where N
    R = observation_covariance(p.x, model, t, data, θ)
    y .= observation_function(p.x, model, t, data, θ) .+ rand(MvNormal(R))
    nothing
end

function log_observation_density(p::FloatParticle{N}, model::SSM, t::Integer, data, θ) where N
    R = observation_covariance(p.x, model, t, data, θ)
    logpdf(MvNormal(R), data.y[t] .- observation_function(p.x, model, t, data, θ))
end

function simulate_transition!(pnext::FloatParticle{N}, pcurr::FloatParticle{N}, model::SSM, t::Integer, data, θ) where N
    Q = transition_covariance(pcurr.x, model, t, data, θ)
    pnext.x .= transition_function(pcurr.x, model, t, data, θ) .+ rand(MvNormal(Q))
    nothing
end

function log_transition_density(pnext::FloatParticle{N}, pcurr::FloatParticle{N}, model::SSM, t::Integer, data, θ) where N
    Q = transition_covariance(pcurr.x, model, t, data, θ)
    logpdf(MvNormal(Q), pnext.x .- transition_function(pcurr.x, model, t, data, θ))
end

function simulate_initial!(p::FloatParticle{N}, model::SSM, data, θ) where N
    μ0 = initial_mean(model, data, θ)
    Σ0 = initial_covariance(model, data, θ)
    p.x .= rand(MvNormal(μ0, Σ0))
    nothing
end

function log_initial_density(p::FloatParticle{N}, model::SSM, data, θ) where N
    μ0 = initial_mean(model, data, θ)
    Σ0 = initial_covariance(model, data, θ)
    logpdf(MvNormal(μ0, Σ0), p.x)
end