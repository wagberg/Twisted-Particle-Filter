using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using SequentialMonteCarlo
using StatsPlots
using ProgressBars
using Random
using DataFrames

Random.seed!(2);

##
nx = 3
nu = 1
ny = 1
T = 100
M = [10, 50, 100, 500]
N_mc = 200
# M_tpf = [10, 50, 100]
M_tpf = M

# Get default parameter values
θ = LGSSMParameter{nx, nu, ny}()

model = LGSSM{nx}()
# generate input signal and allocate for output
data = (u = [randn(nu) for i in 1:T], y = [zeros(ny) for i in 1:T], ks=KalmanStorage(model, T))

# Simulate the model storing the output in data
simulate!(data.y, model, data, θ)

# Preallocate storage for the Kalman filter
ekf!(data.ks, model, data, θ)
smooth!(data.ks, model, data, θ)

# Preallocate storage for the particle filter
pf_storage = ParticleStorage(model, maximum(M), T)

# Estimate the likelihood 40 times for each number of particles in M  using the particle filter
ll = zeros(length(M), N_mc)
for i in eachindex(M)
    println(M[i], " particles")
    for j in ProgressBar(1:size(ll,2))
        ll[i, j] = bpf!(pf_storage, model, data, θ; n_particles = M[i])
    end
end

df = DataFrame(ll')
names!(df, Symbol.(M))
df = stack(df)
insertcols!(df, 2, method=["Bootstrap" for i = 1:size(df, 1)])

ll_tpf = zeros(length(M_tpf), N_mc)
for i in eachindex(M_tpf)
    println("Twisted particle filter with ", M_tpf[i], " particles")
    for j in ProgressBar(1:size(ll_tpf, 2))
        ll_tpf[i, j] = tpf!(pf_storage, model, data, θ; n_particles = M_tpf[i])
    end
end

df_tpf = DataFrame(ll_tpf')
names!(df_tpf, Symbol.(M_tpf))
df_tpf = stack(df_tpf)
insertcols!(df_tpf, 2, method=["Twisted" for i = 1:size(df_tpf, 1)])

append!(df, df_tpf)
# @df df violin(string.(:method), :value, title="Likelihood estimates bootstrap PF", legend=false)
p1 = @df filter(row -> row[:method] == "Bootstrap", df) violin(string.(:variable), :value, side=:left, label="Bootstrap", legend=false)
@df filter(row -> row[:method] == "Twisted", df) violin!(string.(:variable), :value, side=:right, label="Twisted")
# boxplot!(ll_tpf', legend=false, xticks=(1:length(M), string.(M_tpf)))
hline!([data.ks.log_likelihood[end]], color="black", width=2, ls=:dash)

p2 = boxplot(ll_tpf', title="Likelihood estimates twisted PF", legend=false, xticks=(1:length(M), string.(M_tpf)), xlabel="Number of particles")
hline!([data.ks.log_likelihood[end]], color="black", width=2, ls=:dash)

plot(p1, p2)