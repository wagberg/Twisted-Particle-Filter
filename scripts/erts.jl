using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using BenchmarkTools
using GaussianFilters
using CSV
using DataFrames
using LinearAlgebra
using StatsPlots
using Random
using Distributions
using ProgressBars
using StatsFuns
using ThreadsX

import GaussianFilters.measure
import GaussianFilters.jacobian
import GaussianFilters.predict

include(projectdir("src","resampling.jl"))
## Redfine update and run_fitler 
"""
    run_filter(filter::AbstractFilter, b0::GaussianBelief, action_history::Vector{AbstractVector}, measurement_history::Vector{AbstractVector})

Given an initial __predictive__ belief  `b0`, matched-size arrays for action and measurement
histories and a filter, update the beliefs using the filter, and return a
vector of all beliefs and the log likelihood.
"""
function run_filter(filter::AbstractFilter, bp::GaussianBelief, action_history::Vector{A},
    measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}

    # assert matching action and measurement sizes
    @assert length(action_history) == length(measurement_history)

    # initialize belief vector
    beliefs = Vector{GaussianBelief}()

    log_likelihood = 0
    # iterate through and update beliefs
    for (u, y) in zip(action_history, measurement_history)
        bf, ll = measure(filter, bp, y; u = u)
        log_likelihood += ll
        push!(beliefs, bf)
        bp = predict(filter, bf, u)
    end

    return beliefs, log_likelihood
end

"""
    likelihood(filter::AbstractFilter, bf::GaussianBelief, action_history::Vector{AbstractVector}, measurement_history::Vector{AbstractVector})

Given an initial __filtering__ belief  `bf`, matched-size arrays for action and measurement
histories and a filter, returns the log likelihood.
"""
function likelihood(filter::AbstractFilter, bf::GaussianBelief, action_history::Vector{A},
    measurement_history::Vector{B}) where {A<:AbstractVector, B<:AbstractVector}

    bp = predict(filter, bf, action_history[1])

    log_likelihood = 0

    for (u,y) in zip(action_history[2:end], measurement_history[2:end])
        bf, ll = measure(filter, bp, y; u=u)
        log_likelihood += ll
        bp = predict(filter, bf, u)
    end

    return log_likelihood
end

function simulation(filter::AbstractFilter, b0::GaussianBelief,
    action_sequence::Vector{<:AbstractArray},
    rng::AbstractRNG = Random.GLOBAL_RNG)

    # make initial state
    s0 = rand(rng, b0)

    # simulate action sequence
    state_history = [s0]
    measurement_history = Vector{AbstractVector{typeof(s0[1])}}()
    for u in action_sequence
        yn = measure(filter.o, state_history[end], u, rng)
        push!(measurement_history, yn)
        xn = predict(filter.d, state_history[end], u, rng)
        push!(state_history, xn)
    end
    pop!(state_history)
    
    return state_history, measurement_history
end

function measure(filter::ExtendedKalmanFilter, bp::GaussianBelief, y::AbstractVector{a};
    u::AbstractVector{b} = [false]) where {a<:Number, b<:Number}

    # Measurement update
    yp = measure(filter.o, bp.μ, u)
    H = jacobian(filter.o, bp.μ, u)

    # Likelihood
    S = H * bp.Σ * H' + filter.o.V
    ll = logpdf(MvNormal(yp,S), y)

    # Kalman Gain
    # K = bp.Σ * H' * inv(S)
    K = (S \ (H*bp.Σ))'

    # Measurement update
    μn = bp.μ + K * (y - yp)
    Σn = (I - K * H) * bp.Σ
    return GaussianBelief(μn, Σn), ll
end

function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end

# SMOOTHER
abstract type AbstractSmoother end

struct ExtendedRtsSmoother <: AbstractSmoother
    d::DynamicsModel
    o::ObservationModel
    f::ExtendedKalmanFilter

    function ExtendedRtsSmoother(d::DynamicsModel, o::ObservationModel)
        f = ExtendedKalmanFilter(d, o)
        new(d, o, f)
    end
end

function smooth(smoother::ExtendedRtsSmoother, bs::GaussianBelief,
                bf::GaussianBelief, u::AbstractVector{<:Number} = [false])
    bp = predict(smoother.f, bf, u)
    F = jacobian(smoother.f.d, bf.μ, u)
    G = (bp.Σ \ (bf.Σ * F))'
    μ = bf.μ + G * (bs.μ - bp.μ)
    Σ = bf.Σ + G * (bs.Σ - bp.Σ) * G'
    return GaussianBelief(μ, Σ)
end

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

##
df = dropmissing(DataFrame(CSV.File(datadir("exp_raw","CascadedTanksFiles","dataBenchmark.csv"))), :uEst)
Ts = df[:Ts][1]
select!(df, Not(6))
select!(df, Not(:Ts))
dropmissing!(df, disallowmissing=true)

uVal = [[x] for x in df.uVal]
yVal = [[x] for x in df.yVal]
##

θ = [   0.054115135099972;
        0.069031758496706;
        0.044688311520571;
        0.224558186566615;
        -0.002262430348693;
        -0.006535014001622;
        0.002513535972694;
        0.013531902670792;
        6.225503938785832;
        4.9728]

##
function step(x, u)
    k₁, k₂, k₃, k₄, k₅, k₆, σₑ², σᵥ², ξ₁, ξ₂ = θ
    xp = clamp.(x, 0, 10)
    xp[1] += Ts*(-k₁*sqrt(xp[1]) - k₅*xp[1] + k₃*u[1])
    xp[2] += Ts*(k₁*sqrt(xp[1]) + k₅*xp[1] - k₂*sqrt(xp[2]) - k₆*xp[2] + k₄*max.(x[1]-10.0, 0.0))

    return xp
end

W = θ[7]*Matrix{Float64}(I, 2, 2)

dmodel = NonlinearDynamicsModel(step, W)

##
function observe(x, u)
    return clamp.(x[2:2], 0, 10)
end

V = θ[8]*Matrix{Float64}(I, 1, 1)

omodel = NonlinearObservationModel(observe, V)

##
ekf = ExtendedKalmanFilter(dmodel, omodel)

##
function bpf(θ, y, u, M, rng=Random.GLOBAL_RNG)
    initial_distribtuion = MvNormal(θ[9:10], 0.2*Matrix(I,2,2))
    ξ = [rand(initial_distribtuion) for i = 1:M]
    w = fill(-log(M),M)
    ll = 0
    data_residual_distribution = MvNormal(omodel.V)
    for (_y, _u) in zip(y, u)
        # weight
        # w = map((x,w) -> w + logpdf(MvNormal(omodel.V),(_y-measure(omodel, x, _u))), ξ, w)
        map!((x,w) -> w + logpdf(data_residual_distribution,(_y - measure(omodel, x, _u))), w, ξ, w)
        wn = logsumexp(w)
        a = resample_multinomial(exp.(w .- wn), M)
        ll += wn
    
        # propagate
        w = fill(-log(M), M)
        ξ = map!(x->predict(dmodel, x, _u, rng), ξ, ξ[a])
    end
    return ll
end

##
function tpf(θ, y::Vector{<:AbstractVector}, u::Vector{<:AbstractVector}, M::Integer,
    filter::AbstractFilter, look_ahead::Integer, rng=Random.GLOBAL_RNG)
    initial_belief = GaussianBelief(θ[9:10], 0.2*Matrix(I,2,2))
    initial_distribtuion = MvNormal(initial_belief.μ, initial_belief.Σ)
    ξ = [rand(initial_distribtuion) for i = 1:M]
    w = fill(-log(M),M)
    ll = 0
    log_potential = zeros(M)
    data_residual_distribution = MvNormal(omodel.V)
    x_size = size(initial_belief.Σ)

    for t = 1:size(y,1)
        # weight
        map!((x,w,lp) -> w - lp + logpdf(data_residual_distribution,(y[t] - measure(omodel, x, u[t]))), w, ξ, w, log_potential)
        tend = Int(min(t+look_ahead, length(u)))
        map!(x->likelihood(filter, GaussianBelief(x, zeros(x_size)), u[t:tend], y[t:tend]), log_potential, ξ)
        w += log_potential
        wn = logsumexp(w)
        a = resample_multinomial(exp.(w .- wn), M)
        ll += wn
        w = fill(-log(M), M)
        log_potential = log_potential[a]

        # propagate
        map!(x->predict(dmodel, x, u[t], rng), ξ, ξ[a])
    end
    return ll
end

@btime tpf(θ, y, u, 10, ekf, 40)

##
function tpf(θ, y::Vector{<:AbstractVector}, u::Vector{<:AbstractVector}, M::Integer,
    smoother::AbstractSmoother, look_ahead::Integer, rng=Random.GLOBAL_RNG)
    initial_belief = GaussianBelief(θ[9:10], 0.2*Matrix(I,2,2))
    initial_distribtuion = MvNormal(initial_belief.μ, initial_belief.Σ)
    smoothed_beliefs, ll = run_smoother(smoother, initial_belief, u, y)

    ξ = [rand(initial_distribtuion) for i = 1:M]
    w = fill(-log(M),M)
    ll = 0
    log_potential = zeros(M)
    data_residual_distribution = MvNormal(omodel.V)
    x_size = size(initial_belief.Σ)

    for t = 1:size(y,1)
        # weight
        map!((x,w,lp) -> w - lp + logpdf(data_residual_distribution,(y[t] - measure(omodel, x, u[t]))), w, ξ, w, log_potential)
        tend = Int(min(t+look_ahead, length(u)))
        pmap!(x->likelihood(filter, GaussianBelief(x, zeros(x_size)), u[t:tend], y[t:tend]), log_potential, ξ)
        w += log_potential
        wn = logsumexp(w)
        a = resample_multinomial(exp.(w .- wn), M)
        ll += wn
        w = fill(-log(M), M)
        log_potential = log_potential[a]

        # propagate
        map!(x->predict(dmodel, x, u[t], rng), ξ, ξ[a])
    end
    return ll
end

##
y = yVal[1:end]
u = uVal[1:end]
Random.seed!(0)
##

ll_hat = []
K = 1
# for M = [10, 50, 100, 500, 1000]
for M = [100, 500]
    println(M)
    # for _ in ProgressBar(1:K)
    #     push!(ll_hat, Dict(:ll=>bpf(θ, y, u, M), :Particles=>M, :Algorithm=>"Bootstrap"))
    # end
    for _ in ProgressBar(1:K)
        push!(ll_hat, Dict(:ll=>tpf(θ, y, u, M, ekf, 40), :Particles=>M, :Algorithm=>"Twisted"))
    end

end
dfll = DataFrame(ll_hat)

CSV.write("likelihood_estimates_3.csv", dfll)

##
@df dfll boxplot(:Particles, :ll, group=(:Algorithm))

# ##
# smoother = ExtendedRtsSmoother(dmodel, omodel)
# M = 10
# look_ahead = 20
# ##

# initial_belief = GaussianBelief(θ[9:10], 0.2*Matrix(I,2,2))
# initial_distribution = MvNormal(initial_belief.μ, initial_belief.Σ)
# smoothed_beliefs, ll = run_smoother(smoother, initial_belief, u, y)
# initial_proposal = MvNormal(smoothed_beliefs[1].μ, smoothed_beliefs[1].Σ)

# ξ = [rand(initial_proposal) for i = 1:M]
# w = [logpdf(initial_distribution, x) - logpdf(initial_proposal, x) for x in ξ]
# w = fill(-log(M),M)
# ll = 0
# log_potential = zeros(M)
# log_transition = [logpdf(initial_distribution, x) for x in ξ]
# log_proposal = [logpdf(initial_proposal, x) for x in ξ]
# data_residual_distribution = MvNormal(omodel.V)
# x_size = size(initial_belief.Σ)
# for t = 1:size(y,1)
#     # weight
#     map!((x,w,lf,lq,lp) -> w + lf - lq - lp + logpdf(data_residual_distribution,(y[t] - measure(omodel, x, u[t]))),
#         w, ξ, w, log_transition, log_proposal, log_potential)
#     tend = Int(min(t+look_ahead, length(u)))

#     out = [run_smoother_from_filtering(smoother, GaussianBelief(x, zeros(x_size)) , u[t:tend], y[t:tend]) for x in ξ]
#     log_potential = [x[2] for x in out]
#     w += log_potential
#     wn = logsumexp(w)
#     a = resample_multinomial(exp.(w .- wn), M)
#     ll += wn
#     w = fill(-log(M), M)
#     log_potential = log_potential[a]
    
#     xprime = [rand(rng, MvNormal(x[1].μ, x[1].Σ)) for x in out[a]]
#     log_proposal = [logpdf(MvNormal(x[1].μ, x[1].Σ), xx) for (x, xx) in zip(out[a], xprime[a])]
#     log_transition = [logpdf(MvNormal(dmodel.W) , xp-predict(dmodel, x, u[t])) for (xp,x) in zip(xprime[a], ξ[a])]
#     ξ = xprime[a]
# end