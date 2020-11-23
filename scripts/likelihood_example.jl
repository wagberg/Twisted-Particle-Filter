using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using GaussianSmoothers
using BenchmarkTools
using Plots
using LinearAlgebra
using Distributions
using Random
using StaticArrays
using ForwardDiff

Random.seed!(10);

##
# nonlinear dynamics function. must be a function of both states (x) and actions (u) even if either are not used.
dt = 0.001
J = [1; 5; 5]
function step(x,u,w)
    xp = x + dt*u./J + w
    xp[1] += dt*(J[2]-J[3])/J[1]*x[2]*x[3]
    xp[2] += dt*(J[3]-J[1])/J[2]*x[3]*x[1]
    xp[3] += dt*(J[1]-J[2])/J[3]*x[1]*x[2]
    return xp
end    

Q = 0.001*Matrix{Float64}(I,3,3)

# build dynamics model
dmodel = NonlinearDynamicsModel(step,MvNormal(Q));
##
# nonlinear observation function. must be a function of both states (x) and actions (u) even if either are not used.
c=10
function observe(x,u, v)
    y = x
    y = min.(y,c)
    y = max.(y,-c) + v
    return y
end

R = 0.3*Matrix{Float64}(I,3,3)

# build observation model
omodel = NonlinearObservationModel(observe,MvNormal(R))

# build ekf
ekf = ExtendedKalmanFilter(dmodel,omodel);

##
times = 0:dt:5
action_sequence = [[0.0,0.0,0.0] for t in times]

b0 = GaussianBelief([10.0,0.0,0.0], Matrix{Float64}(I,3,3))

sim_states, sim_measurements = simulate(ekf,b0,action_sequence);

##
@btime likelihood(ekf, b0, action_sequence, sim_measurements);
@btime run_filter(ekf, b0, action_sequence, sim_measurements);ew2r43r324r23l mfr m frlmglkrjgerjgithli,4euhg.wh