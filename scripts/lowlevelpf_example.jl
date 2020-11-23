using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using BenchmarkTools
using Parameters
using LowLevelParticleFilters
using LinearAlgebra
using StaticArrays
using Distributions
using Plots

## Define problem

n = 1   # Dimension of state
m = 0   # Dimension of input
p = 1   # Dimension of measurements
N = 5000 # Number of particles
T = 1000

df = MvNormal(n, 1.0)          # Dynamics noise Distribution
d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

@with_kw struct Para
    ϕ = 0.98
    σ = 0.16
    β = 0.70
end
θ = Para()

function dynamics(x,u,t,noise=false) # It's important that this defaults to false
    x = θ.ϕ .* x # A simple dynamics model
    if noise
        x += rand(MvNormal(1, θ.σ))
    end
    x
end
# The `measurement_likelihood` function must have a method accepting state, measurement and time, and returning the log-likelihood of the measurement given the state, a simple example below:
function measurement_likelihood(x,y,t)
    logpdf(MvNormal(1, θ.β .* exp(.5*x[1])), y) # A simple linear measurement model with normal additive noise
end
# This gives you very high flexibility. The noise model in either function can, for instance, be a function of the state, something that is not possible for the simple `ParticleFilter`
# To be able to simulate the `AdvancedParticleFilter` like we did with the simple filter above, the `measurement` method with the signature `measurement(x,t,noise=false)` must be available and return a sample measurement given state (and possibly time). For our example measurement model above, this would look like this
measurement(x,t,noise=false) = noise*rand(MvNormal(1, θ.β .* exp(.5*x[1])))
# We now create the `AdvancedParticleFilter` and use it in the same way as the other filters:

##
apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0)
xs,u,y = simulate(apf, T, df)

x,w,we,ll = forward_trajectory(apf, u, y)
trajectorydensity(apf, x, we, y, xreal=xs)