using ForwardDiff: dualize, Tag, value, extract_gradient, extract_gradient!,
    extract_jacobian, extract_jacobian!, partials

function gradient(pf::AbstractParticleFilter, θ, v2p=v->vec_to_par(_model(pf), v), p2v=par_to_vec)
    n_particles = _n_particles(pf)
    model = _model(pf)
    data = _data(pf)
    run_filter!(pf, θ)
    # Convert prameter to vector
    p = p2v(θ)
    dₚ = length(p)
    # Construct dual variables
    T = typeof(Tag(:log_likelohood, eltype(p)))
    d = dualize(T, p)
    # Get back dual paramter
    th = v2p(d)

    # Pre-allocate
    αₜ = zeros(dₚ,n_particles)
    γₜ = zeros(dₚ,n_particles)
    # Prellocate output
    ∇ = zeros(dₚ)

    # gradient for initial state
    x₁,_ = states(pf, 1)
    for i ∈ eachindex(x₁)
        # gradient of iniital density
        fd = log_initial_density(x₁[i], model, data, th)
        grad = extract_gradient(T, fd, p)
        γₜ[:,i] .= grad

        # gradient of first observation
        fd = log_observation_density(x₁[i], model, 1, data, th)
        grad = extract_gradient(T, fd, p)
        @views γₜ[:,i] .+= grad
    end
    logw₁,_ = weights(pf, 1)
    vₜ = γₜ * exp.(logw₁)
    # w₁ = exp.(logw₁)
    # vₜ = dropdims(sum(γₜ .* transpose(w₁), dims=2), dims=2)
    αₜ .= γₜ .- vₜ
    ∇ .= vₜ

    for t in 2:length(data.y)
        xₜ,xₜ₋₁ = states(pf, t)
        aₜ = ancestors(pf, t-1)
        for i ∈ eachindex(xₜ)
            fd = log_transition_density(xₜ[i], xₜ₋₁[aₜ[i]], model, t-1, data, th)
            grad = extract_gradient(T, fd, p)
            @views γₜ[:,i] .= grad .+ αₜ[:,aₜ[i]]

            fd = log_observation_density(xₜ[i], model, t, data, th)
            grad = extract_gradient(T, fd, p)
            @views γₜ[:,i] .+= grad
        end
        logwₜ,_ = weights(pf, t)
        vₜ = γₜ * exp.(logwₜ)
        αₜ .= γₜ .- vₜ
        ∇ .+= vₜ
    end
    return ∇
end

function extract_grad_and_hess(T, x, v)
    d = length(v)
    return zero(v), zeros(eltype(v),d,d)
end

function extract_grad_and_hess(T, x::ForwardDiff.Dual, v)
    grad = ForwardDiff.extract_gradient(T, ForwardDiff.value(T, x), v)
    hess = ForwardDiff.extract_jacobian(T, ForwardDiff.partials(T, x), v)
    return grad, hess
end

function grad_and_hess(pf::AbstractParticleFilter, θ, v2p=v->vec_to_par(_model(pf), v), p2v=par_to_vec)
    n_particles = _n_particles(pf)
    model = _model(pf)
    data = _data(pf)
    run_filter!(pf, θ)
    # Convert prameter to vector
    p = p2v(θ)
    dₚ = length(p)
    # Construct dual variables
    T = typeof(Tag(:log_likelohood, eltype(p)))
    d1 = dualize(T, p)
    d2 = dualize(T, d1)
    # Get back dual paramter
    th = v2p(d2)

    # Pre-allocate
    αₜ = zeros(dₚ,n_particles)
    βₜ = zeros(dₚ,dₚ,n_particles)
    ϕₜ = zeros(dₚ,dₚ,n_particles)
    γₜ = zeros(dₚ,n_particles)
    # Prellocate output
    ∇ = zeros(dₚ)
    ∇² = zeros(dₚ,dₚ)

    # gradient and hessian for initial state
    x₁,_ = states(pf, 1)
    for i ∈ eachindex(x₁)
        # gradient and hessian of iniital density
        fd2 = log_initial_density(x₁[i], model, data, th)

        grad, hess = extract_grad_and_hess(T, fd2, p)
        # val = value(T, value(T,fd2))
        # grad = extract_gradient(T, value(T, fd2), p)
        # hess = extract_jacobian(T, partials(T, fd2), p)
        @views γₜ[:,i] .= grad
        ϕₜ[:,:,i] .= hess

        # gradient and hessian of first observation
        fd2 = log_observation_density(x₁[i], model, 1, data, th)
        grad, hess = extract_grad_and_hess(T, fd2, p)
        # @show val = value(T, value(T,fd2))
        # @show grad = extract_gradient(T, value(T, fd2), p)
        # @show hess = extract_jacobian(T, partials(T,fd2), p)
        γₜ[:,i] .+= grad
        ϕₜ[:,:,i] .+= hess
    end
    logw₁,_ = weights(pf, 1)
    # w₁ = exp.(logw₁)
    # vₜ = dropdims(sum(γₜ .* transpose(w₁), dims=2), dims=2)
    vₜ = γₜ * exp.(logw₁)
    αₜ .= γₜ .- vₜ
    Bₜ = mapreduce(i->exp(logw₁[i]).*(ϕₜ[:,:,i] .+ γₜ[:,i]*γₜ[:,i]'), +, 1:n_particles)
    βₜ = ϕₜ .- Bₜ
    ∇ .= vₜ
    ∇² .= Bₜ

    for t in 2:length(data.y)
        xₜ,xₜ₋₁ = states(pf, t)
        aₜ = ancestors(pf, t-1)
        for i ∈ eachindex(xₜ)
            fd2 = log_transition_density(xₜ[i], xₜ₋₁[aₜ[i]], model, t-1, data, th)
            grad, hess = extract_grad_and_hess(T, fd2, p)
            # val = value(T, value(T,fd2))
            # grad = extract_gradient(T, value(T, fd2), p)
            # hess = extract_jacobian(T, partials(T,fd2), p)
            γₜ[:,i] .= grad .+ αₜ[:,aₜ[i]]
            ϕₜ[:,:,i] .= hess .+ βₜ[:,:,aₜ[i]]

            fd2 = log_observation_density(xₜ[i], model, t, data, th)
            grad, hess = extract_grad_and_hess(T, fd2, p)
            # val = value(T, value(T,fd2))
            # grad = extract_gradient(T, value(T, fd2), p)
            # hess = extract_jacobian(T, partials(T,fd2), p)
            γₜ[:,i] .+= grad
            ϕₜ[:,:,i] .+= hess
        end
        logwₜ,_ = weights(pf, t)
        # wₜ = exp.(logwₜ)
        # vₜ = dropdims(sum(γₜ .* transpose(wₜ), dims=2), dims=2)
        vₜ = γₜ * exp.(logwₜ)
        αₜ .= γₜ .- vₜ
        # Bₜ = mapreduce(i->wₜ[i].*(ϕₜ[:,:,i] .+ γₜ[:,i]*γₜ[:,i]'), +, 1:n_particles)
        Bₜ = mapreduce(i->exp(logwₜ[i]).*(ϕₜ[:,:,i] .+ γₜ[:,i]*γₜ[:,i]'), +, 1:n_particles)
        βₜ = ϕₜ .- Bₜ
        ∇ .+= vₜ
        ∇² .+= Bₜ
    end
    return ∇,∇²
end
