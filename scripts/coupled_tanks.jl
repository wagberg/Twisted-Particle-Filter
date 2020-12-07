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
using ProgressBars

includet(projectdir("src/models/coupled_tanks.jl"))
##
M = [10, 50, 100, 500, 1_000] # Number of particles
M = [10, 100, 1_000] # Number of particles
M_max = 10_000
N_mc = 100 # Number of Monte Carlo runs fo estimating likelihood

θ = TankParameter()
model = CoupledTank()

df = dropmissing(DataFrame(CSV.File(datadir("dataBenchmark.csv"))), :uEst)
Ts = df[:Ts][1]
select!(df, Not(6))
select!(df, Not(:Ts))
dropmissing!(df, disallowmissing=true)

T = size(df, 1)
ps = ParticleStorage(model, M_max, T)
data = (;
    y=[[x] for x in df.yVal],
    u=[[u] for u in df.uVal],
    ks=KalmanStorage(model, T))

ekf!(data.ks, model, data, θ)
smooth!(data.ks, model, data, θ)

@time bpf!(ps, model, data, θ, n_particles = 10_000)
@time tpf!(ps, model, data, θ, n_particles = 1_000)
##

ll = zeros(N_mc, 2*length(M))
for i in eachindex(M)
    println("Running ", M[i]," particles:")
    for j in ProgressBar(1:N_mc)
        ll[j,i] = bpf!(ps, model, data, θ; n_particles=M[i])
        ll[j,length(M)+i] = tpf!(ps, model, data, θ; n_particles=M[i])
    end
end

ll_hat = bpf!(ps, model, data, θ)

##

d = DataFrame(vcat(([hcat(ll[:,i+1], fill(M[mod(i,length(M))+1], size(ll,1)), fill(i>length(M)-1 ? "Twisted" : "Bootstrap", size(ll,1))) for i in 0:2*length(M)-1])...))
names!(d, [:likelihood, :particles, :method])

##
using Gadfly, Cairo, Fontconfig

p = Gadfly.plot(
    d,
    x=:particles,
    y=:likelihood,
    color=:method,
    Gadfly.Scale.x_discrete(levels=M, labels=string),
    Gadfly.Geom.boxplot,
    Gadfly.Theme(boxplot_spacing=0.2*Gadfly.cx),
    Gadfly.Guide.colorkey(title="Method")#, pos=[0.0*Gadfly.w,-0.3*Gadfly.h])
    )

##
p_traj = Gadfly.plot(

)

##
p |> PDF(projectdir("plots", "likelihood_estimated.pdf"))

##
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

