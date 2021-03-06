using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

##
using GaussianSmoothers
using SequentialMonteCarlo
using Parameters
using StaticArrays
using LinearAlgebra
using Distributions
using Random
using StatsPlots
using BenchmarkTools


includet(projectdir("src/models/spinning_satellite.jl"))

Random.seed!(5);
##
θ = SpinningSatelliteParameter()
model = SpinningSatellite()

times = 0:θ.dt:5

data = (u=[[0.0,0.0,0.0] for t in times], y=[zeros(3) for t in times], x=[FloatParticle{3}() for t in times])

simulate!(data.y, data.x, model, data, θ)
storage = KalmanStorage(FloatParticle{3}, length(times))

# Setup model gor GaussianSmoothers to compare the results
# nonlinear dynamics function. must be a function of both states (x) and actions (u) and noise (w) even if either are not used.
dt = θ.dt
J = θ.J
function step(x,u,w)
    xp = x .+ dt*u./J .+ w
    xp[1] += dt*(J[2]-J[3])/J[1]*x[2]*x[3]
    xp[2] += dt*(J[3]-J[1])/J[2]*x[3]*x[1]
    xp[3] += dt*(J[1]-J[2])/J[3]*x[1]*x[2]
    return xp
end    

# nonlinear observation function. must be a function of both states (x) and actions (u) and noise (e) even if either are not used.
c = θ.c
function observe(x,u, e)
    y = x
    y = min.(y,c)
    y = max.(y,-c) + e
    return y
end
b0 = GaussianBelief(θ.μ0, convert(Array, θ.Σ0))

# build observation model
omodel = NonlinearObservationModel(observe,MvNormal(θ.R))
dmodel = NonlinearDynamicsModel(step, MvNormal(θ.Q));
ekf = ExtendedKalmanFilter(dmodel,omodel);
rts = ExtendedRtsSmoother(ekf);


filtered_beliefs, log_likelihood = run_filter(ekf, b0, data.y, u=data.u);
smoothed_beliefs, _ = run_smoother(rts, b0, data.y; u=data.u);

ekf!(storage, model, data, θ)
smooth!(storage, model, data, θ)

## turn array of belief structs into simple tensors.
@assert all([isapprox(storage.filter_mean[t],filtered_beliefs[t].μ) for t in 1:length(filtered_beliefs)])
@assert all([isapprox(storage.filter_Sigma[t],filtered_beliefs[t].Σ) for t in 1:length(filtered_beliefs)])
@assert all([isapprox(storage.smooth_mean[t],smoothed_beliefs[t].μ) for t in 1:length(filtered_beliefs)])
@assert all([isapprox(storage.smooth_Sigma[t],smoothed_beliefs[t].Σ) for t in 1:length(filtered_beliefs)])

println("Benchmarking ekf")
@btime ekf!(storage, model, data, θ);
println("Benchmarking GaussianSmoothers ekf")
@btime run_filter(ekf, b0, data.y; u=data.u);

function new_smooth()
    ekf!(storage, model, data, θ)
    smooth!(storage, model, data, θ)
end
println("Benchmarking GaussianSmoothers rts")
@btime run_smoother(rts, b0, data.y; u=data.u);
println("Benchmarking rts")
@btime new_smooth();

##
x = zeros(3, length(data.x))
for i in eachindex(data.x)
    @inbounds x[:,i] = data.x[i].x
end

μf = hcat(storage.filter_mean...)'


##
plot(
    plot(times,x', label=["x1" "x2" "x3"]),
    plot(times,μf, label=["μf1" "μf2" "μf3"]),
    plot(times,μf, label=["μs1" "μs2" "μs3"])
    )