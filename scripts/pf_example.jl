using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

##
using DataFrames
using StatsFuns
using CSV
using GaussianSmoothers
using LinearAlgebra
using Parameters
using Distributions
using Random
using Plots
using StaticArrays

Random.seed!(1);


##
function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end

##
df = DataFrame(CSV.File(datadir("exp_raw", "seOMXlogreturns2012to2014.csv")))
yVal = [[x] for x in df.log_returns]

abstract type Parameter end

##
@with_kw struct SVParameter <: Parameter
    ϕ = 0.98
    σ = 0.16
    β = 0.70
end
θ = SVParameter()


dmodel = LinearDynamicsModel(fill(θ.ϕ, 1, 1), fill(θ.σ^2, 1, 1))

f(x, u, v, θ) = θ.ϕ.*x + v
dmodel = NonlinearDynamicsModel(f, MvNormal(fill(θ.σ, 1, 1)))

h(x, u, e, θ) = θ.β .* exp.(0.5.*x) .* e

omodel = NonlinearObservationModel(h, MvNormal(fill(1, 1, 1)))

initial_distribtuion = MvNormal(Matrix(I,1,1))

##
ekf = ExtendedKalmanFilter(dmodel, omodel)

b0 = GaussianBelief([0.0], Matrix{Float64}(I,1,1))

xn, yn = simulate(ekf, b0, 200)

filtered_beliefs, ll = run_filter(ekf, b0, yn)
μ, Σ = unpack(filtered_beliefs);

##
# scatter([x[1] for x in yn])
plot(μ[:,1])
plot!([x[1] for x in xn])
##
function bpf(y, M, rng=Random.GLOBAL_RNG)
    ξ = [rand(initial_distribtuion) for i = 1:M]
    w = fill(-log(M),M)
    ll = 0
    for _y in y
        # weight
        # w = map((x,w) -> w + logpdf(MvNormal(omodel.V),(_y-measure(omodel, x, _u))), ξ, w)
        map!((x,w) -> w + logpdf(omodel.d,(_y - measure(omodel, x))), w, ξ, w)
        wn = logsumexp(w)
        a = resample_multinomial(exp.(w .- wn), M)
        ll += wn
    
        # propagate
        w = fill(-log(M), M)
        ξ = map!(x->predict(dmodel, x, rng), ξ, ξ[a])
    end
    return ll
end

##
bpf(yVal, 10000)