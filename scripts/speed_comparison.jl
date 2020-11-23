using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots


n = 2   # Dimension of state
m = 2   # Dimension of input
p = 2   # Dimension of measurements
N = 500 # Number of particles

const dg = MvNormal(p,1.0)          # Measurement noise Distribution
const df = MvNormal(n,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution