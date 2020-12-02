using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using SequentialMonteCarlo
using Distributions
using Parameters
using DataFrames
using CSV
using StatsPlots 

includet(projectdir("src/models/stochastic_volatility.jl"))

M = 200 # Number of particles

θ = SVParameter()
model = StochasticVolatility()

df = DataFrame(CSV.File(datadir("seOMXlogreturns2012to2014.csv")))
data = (;y=[[x] for x in df.log_returns])

ps = ParticleStorage(StochasticVolatilityParticle, M, length(data.y))
ks = KalmanStorage(StochasticVolatilityParticle, length(data.y))

pf!(ps, model, data, θ)
ekf!(ks, model, data, θ)
smooth!(ks, model, data, θ)

X = zeros(size(ps.X))
for i in eachindex(ps.X)
    X[i] = ps.X[i].x[1]
end

xhat = sum.(eachcol(X .* ps.W))

xsig = sum.(eachcol(((X' .- xhat) .^ 2)' .* ps.W))

##

# Kalman

p1 = plot(vcat(ks.filter_mean...), ribbon=2.0 .* sqrt.(vcat(ks.filter_Sigma...)), label="x ± 2σ", title="KF")

p2 = plot(xhat, ribbon=2*xsig, label="x ± 2σ", title="Bootstrap PF")

p3 = plot(zeros(length(data.y)), ribbon=2 .* exp.(θ.logβ .+ vcat(ks.filter_mean...) ./ 2), title="KF", label="ŷ ± 2σ")
@df df scatter!(:log_returns, label="y", ms=4, markershape=:star, msw=0.5)

p4 = plot(zeros(length(data.y)), ribbon = 2 .* exp.(θ.logβ .+ xhat./2), label="ŷ ± 2σ", title="Bootstrap PF")
@df df scatter!(:log_returns, label="y", ms=4, markershape=:star, msw=0.5)

plot(p1, p2, p3, p4)

