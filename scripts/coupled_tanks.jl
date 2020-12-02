using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using SequentialMonteCarlo
using LinearAlgebra
using StaticArrays
using Distributions
using Parameters
using DataFrames
using CSV
using StatsPlots 

includet(projectdir("src/models/coupled_tanks.jl"))
##
M = 200 # Number of particles

θ = TankParameter()
model = CoupledTank()

df = dropmissing(DataFrame(CSV.File(datadir("dataBenchmark.csv"))), :uEst)
Ts = df[:Ts][1]
select!(df, Not(6))
select!(df, Not(:Ts))
dropmissing!(df, disallowmissing=true)

data = (;y=[[x] for x in df.yVal], u=[[u] for u in df.uVal])

ysim = [zeros(1) for t in 1:length(data.u)]
simulate!(ysim, model, data, θ)

ps = ParticleStorage(model, M, length(data.y))
ks = KalmanStorage(model, length(data.y))

ekf!(ks, model, data, θ)
smooth!(ks, model, data, θ)

pf!(ps, model, data, θ)

dx = length(ps.X[1,1].x)
X = zeros(size(ps.X)..., dx)
for i in 1:size(ps.X,1), j in 1:size(ps.X, 2), n in 1:dx
    X[i,j,n] = ps.X[i,j].x[n]
end

xhat = reshape(sum(X .* ps.W; dims=1),:,2)
xsig = reshape(sqrt.(sum((X .- reshape(xhat,1,:,2)) .^ 2 .* ps.W; dims=1)), :, 2)

xhat_kf = transpose(hcat(ks.filter_mean...))
xsig_kf = sqrt.(transpose(hcat(map(diag,ks.filter_Sigma)...)))

##

# Kalman

p1 = plot(kfX[:,1], ribbon= 2 .* xsig_kf[:,1], label="x1 ± 2σ", title="KF")
plot!(kfX[:,2], ribbon= 2 .* xsig_kf[:,2], label="x2 ± 2σ")

p2 = plot(xhat[:,1], ribbon = 2 .* xsig[:,1], label="x1 ± 2±σ", title="Bootstrap PF")
plot!(xhat[:,2], ribbon = 2 .* xsig[:,2], label="x2 ± 2σ")

plot(p1, p2)

