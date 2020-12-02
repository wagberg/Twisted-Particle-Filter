using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

##
using SequentialMonteCarlo
using StatsPlots
using ProgressBars
using Random

Random.seed!(1);

##
nx = 3
nu = 1
ny = 1
T = 100
M = [10, 100, 1_000]

# Get default parameter values
θ = LGSSMParameter{nx, nu, ny}()

model = LGSSM{nx}()
# generate input signal and allocate for output
data = (u = [randn(nu) for i in 1:T], y = [zeros(ny) for i in 1:T])

# Simulate the model storing the output in data
simulate!(data.y, model, data, θ)

# Preallocate storage for the Kalman filter
ekf_storage = KalmanStorage(FloatParticle{3}, T)
ekf!(ekf_storage, model, data, θ)

# Preallocate storage for the particle filter
pf_storage = ParticleStorage(FloatParticle{nx}, maximum(M), T)

# Estimate the likelihood 40 times for each number of particles in M  using the particle filter
ll = zeros(length(M), 40)
for i in eachindex(M)
    println(M[i], " particles")
    for j in ProgressBar(1:size(ll,2))
        ll[i, j] = pf!(pf_storage, model, data, θ; n_particles = M[i])
    end
end

##
boxplot(ll', title="Estimates of the log likelihood", legend=false, xticks=(1:length(M), string.(M)), xlabel="Number of particles")
hline!([ekf_storage.log_likelihood[end]], color="black", width=2, ls=:dash)
