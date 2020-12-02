using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

##
using GaussianSmoothers
using LinearAlgebra
using Distributions
using Random
using Plots
using StaticArrays
using Parameters
using ForwardDiff
using BenchmarkTools

Random.seed!(1);

## Linear Gaussian state-space model

const AVec = AbstractVector;
const AMat = AbstractMatrix;
const AFloat = AbstractFloat;

includet(projectdir("src/statespace.jl"))
includet(projectdir("src/FloatParticle.jl"))
includet(projectdir("src/kalman.jl"))

includet(projectdir("src/models/LGSSM.jl"))

##
nx = 3
nu = 1
ny = 1

dt = 0.1
m = 50
A = [1 dt 0 0 ; 0 1 0 0 ; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt/m 0; 0 0; 0 dt/m]
Q = Symmetric(0.1*Matrix{Float64}(I,4,4))
C = [0 1.0 0 0; 0 0 0 1.0]
D = [0.0 0.0; 0.0 0.0]
R = Symmetric(0.5*Matrix{Float64}(I,2,2))
μ0 = [0.0,0.0,0.0,0.0]
Σ0 = Symmetric(2.0*Matrix{Float64}(I,4,4))

dmodel = LinearDynamicsModel(A,B,Q);
omodel = LinearObservationModel(C,D,R)
θ = LGSSMParameter(SMatrix{size(A)...}(A), SMatrix{size(B)...}(B), SMatrix{size(C)...}(C), SMatrix{size(D)...}(D), Q ,R, SVector{size(μ0)...}(μ0), Σ0)

b0 = GaussianBelief(μ0, Σ0)
kf = KalmanFilter(dmodel,omodel);
rts = RtsSmoother(kf)

times = 0:dt:1000
Fmag = 1000
action_sequence = [[Fmag*cos(t), Fmag*sin(t)] for t in times]

sim_states, sim_measurements = simulate(kf,b0,action_sequence);
filtered_beliefs, log_likelihood = run_filter(kf, b0, sim_measurements; u=action_sequence)
smoothed_beliefs, _ = run_smoother(rts, b0, sim_measurements; u=action_sequence);

data = (y=sim_measurements, u=action_sequence)
model = LGSSM{size(A,1)}()

storage = KalmanStorage(FloatParticle{size(A,1)}, length(times))

ekf!(storage, model, data, θ)
smooth!(storage, model, data, θ)

@assert all([isapprox(storage.filter_mean[t],filtered_beliefs[t].μ) for t in 1:length(filtered_beliefs)])
@assert all([isapprox(storage.filter_Sigma[t],filtered_beliefs[t].Σ) for t in 1:length(filtered_beliefs)])

@assert all([isapprox(storage.smooth_mean[t],smoothed_beliefs[t].μ) for t in 1:length(filtered_beliefs)])
@assert all([isapprox(storage.smooth_Sigma[t],smoothed_beliefs[t].Σ) for t in 1:length(filtered_beliefs)])

@btime ekf!(storage, model, data, θ);
@btime run_filter(kf, b0, sim_measurements; u=action_sequence);

function new_smooth()
    ekf!(storage, model, data, θ)
    smooth!(storage, model, data, θ)
end

@btime run_smoother(rts, b0, sim_measurements; u=action_sequence);
@btime new_smooth();

