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

Random.seed!(1);

##
dt = 0.1
m = 50
A = [1 dt 0 0 ; 0 1 0 0 ; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt/m 0; 0 0; 0 dt/m]
Q = 0.1*Matrix{Float64}(I,4,4)

dmodel = LinearDynamicsModel(A,B,Q);

##
C = [0 1.0 0 0; 0 0 0 1.0]
D = [0.0 0.0; 0.0 0.0]
R = 0.5*Matrix{Float64}(I,2,2)

omodel = LinearObservationModel(C,D, R)





## 
kf = KalmanFilter(dmodel,omodel);


##
b0 = GaussianBelief([0.0,0.0,0.0,0.0], 2.0*Matrix{Float64}(I,4,4))

times = 0:dt:10
Fmag = 1000
action_sequence = [[Fmag*cos(t), Fmag*sin(t)] for t in times]

sim_states, sim_measurements = simulate(kf,b0,action_sequence);

##
filtered_beliefs, log_likelihood = run_filter(kf, b0, action_sequence, sim_measurements)

# turn array of belief structs into simple tensors.
μ, Σ = unpack(filtered_beliefs;dims=[1,3]);

##
x = [x[1] for x in sim_states]
y = [x[3] for x in sim_states]
plot(x, y)
plot!(μ[:,1], μ[:,2])

for i in 1:20:size(μ,1)
    x,y = belief_ellipse(μ[i,:], Σ[i,:,:])
    plot!(x,y)
end
plot!(legend=false)