### SIMULATION FUNCTIONS ###

"""
    simulate_step(filter::AbstractFilter, x::AbstractVector, u::AbstractVector, rng::AbstractRNG=Random.GLOBAL_RNG)

Run a step of simulation starting at state x, taking action u, and using the
motion and measurement equations specified by the filter.
"""
function simulate_step(filter::AbstractFilter, x::AbstractVector{<:Number},
        u::AbstractVector{<:Number}, rng::AbstractRNG=Random.GLOBAL_RNG)
    
    yn = measure(filter.o, x, rng; u=u)
    xn = predict(filter.d, x, rng; u=u)
    
    return xn, yn
end

function simulate_step(filter::AbstractFilter, x::AbstractVector{<:Number}, rng::AbstractRNG=Random.GLOBAL_RNG)

    yn = measure(filter.o, x, rng)
    xn = predict(filter.d, x, rng)
    
    return xn, yn
    end

"""
    simulation(filter::AbstractFilter, b0::GaussianBelief,
                action_sequence::Vector{AbstractVector}})

Run a simulation to get positions and measurements. Samples starting point from
GaussianBelief b0, the runs action_sequence with additive gaussian noise all
specified by AbstractFilter filter to return a simulated state and measurement
history.
"""
function simulate(filter::AbstractFilter, b0::GaussianBelief,
                    action_sequence::Vector{<:AbstractArray},
                    rng::AbstractRNG = Random.GLOBAL_RNG)

    # sample initial state
    xn = rand(rng, b0)

    # simulate action sequence
    state_history = Vector{AbstractVector{typeof(xn[1])}}()
    measurement_history = Vector{AbstractVector{typeof(xn[1])}}()
    for u in action_sequence
        push!(state_history, xn)
        xn, yn = simulate_step(filter, state_history[end], u, rng)
        push!(measurement_history, yn)
    end

    return state_history, measurement_history
end

function simulate(filter::AbstractFilter, b0::GaussianBelief, N::Integer, rng::AbstractRNG = Random.GLOBAL_RNG)
    # sample initial state
    xn = rand(rng, b0)

    # simulate action sequence
    state_history = Vector{AbstractVector{typeof(xn[1])}}()
    measurement_history = Vector{AbstractVector{typeof(xn[1])}}()
    for u in range(0,length=N)
        push!(state_history, xn)
        xn, y = simulate_step(filter, state_history[end], rng)
        push!(measurement_history, y)
    end

    return state_history, measurement_history
end

"""
likelihood(filter::AbstractFilter, bp::GaussianBelief, action_history::Vector{AbstractVector},
            measurement_history::Vector{AbstractVector})

Given an initial __predictive__ belief bp, matched-size arrays for action and measurement
histories and a filter, update the beliefs using the filter, and return a
vector of all beliefs.
"""
function likelihood(filter::AbstractFilter, bp::GaussianBelief, action_history::Vector{A},
            measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}

        # assert matching action and measurement sizes
        @assert length(action_history) == length(measurement_history)

        # initialize belief vector
        log_likelihood = 0
        # iterate through and update beliefs
        for (u, y) in zip(action_history, measurement_history)
            bf, ll = measure(filter, bp, y, u)
            log_likelihood += ll
            bp = predict(filter, bf, u)
        end

        return log_likelihood
end


"""
    run_filter(filter::AbstractFilter, bp::GaussianBelief, action_history::Vector{AbstractVector},
            measurement_history::Vector{AbstractVector})

Given an initial __predictive__ belief bp, matched-size arrays for action and measurement
histories and a filter, update the beliefs using the filter, and return a
vector of all beliefs.
"""
function run_filter(filter::AbstractFilter, bp::GaussianBelief, y::Vector{B}; u::Vector{A} = [zeros(0) for _ in y]) where {A<:AbstractVector, B<:AbstractVector}

        # assert matching action and measurement sizes
        @assert length(u) == length(y)

        # initialize belief vector
        beliefs = Vector{GaussianBelief}()

        log_likelihood = 0
        # iterate through and update beliefs
        for (_u, _y) in zip(u, y)
            bf, ll = measure(filter, bp, y; u=_u)
            log_likelihood += ll
            push!(beliefs, bf)
            bp = predict(filter, bf; u=_u)
        end

        return beliefs, log_likelihood
end

# function run_filter(filter::AbstractFilter, b0::GaussianBelief, measurement_history::Vector{T}) where {T<:AbstractVector}
#     return run_filter(filter, b0, [zeros(0) for _ in measurement_history], measurement_history)
# end

"""
    run_smoother(smoother::AbstractSmoother, b0::GaussianBelief, action_history::Vector{AbstractVector}, measurement_history::Vector{AbstractVector})

Given an initial __predictive__ belief  `b0`, matched-size arrays for action and measurement
histories and a filter, update the beliefs using the filter, run a backwards smoothing
sweep, and return a vector of all smoothed beliefs and the log likelihood.
"""
function run_smoother(smoother::AbstractSmoother, b0::GaussianBelief, action_history::Vector{A},
    measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}
       
    filter_beliefs, ll = run_filter(smoother.f, b0, action_history, measurement_history)

    bs = predict(smoother.f, filter_beliefs[end], action_history[end])
    smoothed_beliefs = Vector{GaussianBelief}()
    for (u, bf) in zip(reverse(action_history), reverse(filter_beliefs))
        bs = smooth(smoother, bs, bf, u)
        pushfirst!(smoothed_beliefs, bs)
    end
    return smoothed_beliefs, ll
end

"""
    run_smoother(smoother::AbstractSmoother, bf::GaussianBelief, action_history::Vector{AbstractVector}, measurement_history::Vector{AbstractVector})

Given an initial __filtering__ belief  `bf`, matched-size arrays for action and measurement
histories and a filter, update the beliefs using the filter, run a backwards smoothing
sweep, and return a vector of all smoothed beliefs and the log likelihood. The log likelihood
is p(y_{2:T} | x_{1}).
"""
function run_smoother_from_filtering(smoother::AbstractSmoother, bf::GaussianBelief, action_history::Vector{A},
    measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}
    
    bp = predict(smoother.f, bf, action_history[1])
    
    filter_beliefs, ll = run_filter(smoother.f, bp, action_history[2:end], measurement_history[2:end])

    bs = predict(smoother.f, filter_beliefs[end], action_history[end])
    # smoothed_beliefs = Vector{GaussianBelief}()
    for (u, bf) in zip(reverse(action_history[2:end]), reverse(filter_beliefs[2:end]))
        bs = smooth(smoother, bs, bf, u)
        # pushfirst!(smoothed_beliefs, bs)
    end
    return bs, ll
end


"""
    unpack(belief_history::Vector{<:GaussianBelief};
        dims::Vector{Int}=[])

Given a history of beliefs, return an unpacked (time steps, state dim)-sized array of
predicted means and a (time steps, state dim, state dim)-sized array of
covariances. One can optionally specify dimensions indices dims to output
reduced state information.
"""
function unpack(belief_history::Vector{<:GaussianBelief};
    dims::Vector{Int}=Vector{Int}())

    # set default to condense all dimensions
    if length(dims) == 0
        dims = collect(1:length(belief_history[1].μ))
    end

    # set output sizes
    μ = zeros(typeof(belief_history[1].μ[1]),
        length(belief_history), length(dims))
    Σ = zeros(typeof(belief_history[1].μ[1]),
        length(belief_history), length(dims), length(dims))

    # iterate over belief_history and place elements appropriately
    for (i, belief) in enumerate(belief_history)
        μ[i,:] = belief.μ[dims]
        Σ[i,:,:] = belief.Σ[dims,dims]
    end

    return μ, Σ
end

