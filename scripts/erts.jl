using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using BenchmarkTools
using GaussianSmoothers
using CSV
using DataFrames
using LinearAlgebra
using StatsPlots
using Random
using Distributions
using ProgressBars
using StatsFuns
using ThreadsX

include(projectdir("src","resampling.jl"))

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
struct TankDynamicsModel <: DynamicsModel
    d::Distribution
end
function GaussianSmoothers.predict(m::TankDynamicsModel, x::AbstractVector{<:Number}, 
    u::AbstractVector{<:Number}, w::AbstractVector{<:Number})
    k₁, k₂, k₃, k₄, k₅, k₆, σₑ², σᵥ², ξ₁, ξ₂ = θ
    xp = clamp.(x, 0, 10)
    xp[1] += Ts*(-k₁*sqrt(xp[1]) - k₅*xp[1] + k₃*u[1])
    xp[2] += Ts*(k₁*sqrt(xp[1]) + k₅*xp[1] - k₂*sqrt(xp[2]) - k₆*xp[2] + k₄*max.(x[1]-10.0, 0.0))

    return xp .+ w
end
function GaussianSmoothers.jacobian(::GaussianSmoothers.NoiseJacobian, m::TankDynamicsModel, x; u=zeros(eltype(x),0))
    Array{eltype(x), 2}(I, length(x), length(x))
end
function GaussianSmoothers.jacobian(::GaussianSmoothers.StateJacobian, m::TankDynamicsModel, x; u=zeros(eltype(x),0))
    k₁, k₂, k₃, k₄, k₅, k₆, σₑ², σᵥ², ξ₁, ξ₂ = θ
    xp = zeros(2,2)
    
    if x[1] < 10.0 && x[1] > 0
        xp[1,1] = 1.0 + Ts*(-0.5*k₁/sqrt(x[1]) - k₅)
        xp[2,1] = Ts*(0.5*k₁/sqrt(x[1]) + k₅)
    end
    if x[2] < 10.0 && x[2] > 0
        xp[2,2] = 1.0 + Ts*(-0.5*k₂/sqrt(x[2]) - k₆)
    end
    if x[1] > 10.0
        xp[2,1] += Ts*k₄
    end

    return xp
end

W = θ[7]*Matrix{Float64}(I, 2, 2)
tankdmodel = TankDynamicsModel(MvNormal(W))
##
struct TankObservationModel <: ObservationModel
    d::Distribution
end
function GaussianSmoothers.measure(m::TankObservationModel, x::AbstractVector{<:Number}, 
    u::AbstractVector{<:Number}, e::AbstractVector{<:Number})
    clamp.(x[2:2], 0, 10) .+ e
end
function GaussianSmoothers.jacobian(::GaussianSmoothers.NoiseJacobian, m::TankObservationModel, x; u=zeros(eltype(x),0))
    Array{Float64, 2}(I, 1, 1)
end
function GaussianSmoothers.jacobian(::GaussianSmoothers.StateJacobian, m::TankObservationModel, x; u=zeros(eltype(x),0))
    xp = zeros(1,2)
    if x[2] < 10.0 && x[2] > 0
        xp[1,2] = 1.0
    end
    xp
end
function observe(x, u, e)
    return clamp.(x[2:2], 0, 10) .+ e
end

V = θ[8]*Matrix{Float64}(I, 1, 1)

omodel = NonlinearObservationModel(observe, MvNormal(V))
tankomodel = TankObservationModel(MvNormal(V))
##

ekf = ExtendedKalmanFilter(dmodel, omodel)
tankekf = ExtendedKalmanFilter(tankdmodel, tankomodel)
run_filter(ekf, GaussianBelief([5.0, 5.0], W), yVal, u=uVal)
run_filter(tankekf, GaussianBelief([5.0, 5.0], W), yVal, u=uVal)
@btime run_filter(ekf, GaussianBelief([5.0, 5.0], W), yVal, u=uVal);
@btime run_filter(tankekf, GaussianBelief([5.0, 5.0], W), yVal, u=uVal);
##


function TankDynamicsModel(θ)
    let
        k₁, k₂, k₃, k₄, k₅, k₆, σₑ², σᵥ², ξ₁, ξ₂ = θ
        function f(x, u)
            xp = clamp.(x, 0, 10)
            xp[1] += Ts*(-k₁*sqrt(xp[1]) - k₅*xp[1] + k₃*u[1])
            xp[2] += Ts*(k₁*sqrt(xp[1]) + k₅*xp[1] - k₂*sqrt(xp[2]) - k₆*xp[2] + k₄*max.(x[1]-10.0, 0.0))
        
            return xp
        end

        function df(x, u)
            k₁, k₂, k₃, k₄, k₅, k₆, σₑ², σᵥ², ξ₁, ξ₂ = θ
            xp = zeros(2,2)
            
            if x[1] < 10.0 && x[1] > 0
                xp[1,1] = Ts*(1.0 - 0.5*k₁/sqrt(x[1]) + k₅)
                xp[2,1] = Ts*(0.5*k₁/sqrt(x[1]) + k₅)
            end
            if x[2] < 10.0 && x[1] > 0
                xp[2,2] = Ts*(1.0 - 0.5*k₂/sqrt(x[2]) - k₆)
            end
            if x[1] > 10.0
                xp[2,1] += Ts*k₄

            xp[1] += Ts*(-k₁*sqrt(xp[1]) - k₅*xp[1] + k₃*u[1])
            xp[2] += Ts*(k₁*sqrt(xp[1]) + k₅*xp[1] - k₂*sqrt(xp[2]) - k₆*xp[2] + k₄*max.(x[1]-10.0, 0.0))
        
            return xp
        end
        new()
    end
##


##
function bpf(θ, y, u, M, rng=Random.GLOBAL_RNG)
    initial_distribtuion = MvNormal(θ[9:10], 0.2*Matrix(I,2,2))
    ξ = [rand(initial_distribtuion) for i = 1:M]
    w = fill(-log(M),M)
    ll = 0
    data_residual_distribution = omodel.d
    for (_y, _u) in zip(y, u)
        # weight
        # w = map((x,w) -> w + logpdf(MvNormal(omodel.V),(_y-measure(omodel, x, _u))), ξ, w)
        map!((x,w) -> w + logpdf(data_residual_distribution,(_y - measure(omodel, x; u=_u))), w, ξ, w)
        wn = logsumexp(w)
        a = resample_multinomial(exp.(w .- wn), M)
        ll += wn
    
        # propagate
        w = fill(-log(M), M)
        map!(x->predict(dmodel, x, rng; u=_u), ξ, ξ[a])
    end
    return ll
end

@btime bpf(θ, yVal, uVal, 100)
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