using Revise
using DrWatson
@quickactivate "Twisted Particle Filter"

using SequentialMonteCarlo
using StatsPlots
using ProgressBars
using Random
using DataFrames
using LinearAlgebra

Random.seed!(1);

##
nx = 3
nu = 1
ny = 1
T = 50
M = [10, 50, 100, 500]
N_mc = 200
# M_tpf = [10, 50, 100]
M_tpf = M
M_max = 10_000

# Get default parameter values
θ = LGSSMParameter{nx, nu, ny}()

model = LGSSM{nx}()
# generate input signal and allocate for output
data = (u = [randn(nu) for i in 1:T], y = [zeros(ny) for i in 1:T], ks=KalmanStorage(model, T))

P = SequentialMonteCarlo.particletype(model)
ptrue = [P() for t in 1:T]
# Simulate the model storing the output in data
simulate!(data.y, ptrue, model, data, θ) # ptrue is a 1D aprticle, but just used to store the true x-trajectory 

# Preallocate storage for the Kalman filter
ekf!(data.ks, model, data, θ)
smooth!(data.ks, model, data, θ)

# Preallocate storage for the particle filter
bpfs = ParticleStorage(model, M_max, T)
tpfs = ParticleStorage(model, M_max, T)

#ll_bpf = bpf!(bpfs, model, data, θ)
#ll_tpf = tpf!(tpfs, model, data, θ)


##################################################################
# # For testing conditional PF
# Xref = bpf!(bpfs, model, data, θ;  conditional = true, ancestorsampling = true, n_particles = M[1]);
# Xref = bpf!(bpfs, model, data, θ;  conditional = true, n_particles = M[1]);
# println("Done!")

##

x = [[P() for t in 1:T] for n in 1:10_000]

println()
println("Running bpf")
bpf!(bpfs, model, data, θ; n_particles = M[1]); # Run bpf first iteration
SequentialMonteCarlo.condition_on_particle!(bpfs, 1)  # WHY 1 as input here?

for i in ProgressBar(eachindex(x))
    bpf!(bpfs, model, data, θ; n_particles=M[1], conditional=:yes)
    for j in eachindex(x[i])
        SequentialMonteCarlo.copy!(x[i][j], bpfs.X[1,j])
    end
end
##
trueX = hcat([p.x for p in ptrue]...)'
rtsX = hcat(data.ks.smooth_mean...)'
rtsσ = sqrt.(hcat(diag.(data.ks.smooth_Sigma)...))'

##
p1 = plot(trueX[:,1], ls=:dot, label="x_1", title="Posterior samples")
plot!(rtsX[:,1], ribbon=2*rtsσ[:,1], label="xhat_1")
plot!(map(p->p.x[1], x[1]), lw=1, lc=:black, la=0.1, label=false)
for i in 2:100:length(x)
    plot!(map(p->p.x[1], x[i]), lw=1, lc=:black, la=0.1,label=false)
end

plot(p1)

println()
println("Running tpf")
x_tpf = [[P() for t in 1:T] for n in 1:10_000]

tpf!(tpfs, model, data, θ; n_particles = M[1]); # Run bpf first iteration
SequentialMonteCarlo.condition_on_particle!(tpfs, 1)  # WHY 1 as input here?

for i in ProgressBar(eachindex(x_tpf))
    tpf!(tpfs, model, data, θ; n_particles=M[1], conditional=:yes)
    for j in eachindex(x_tpf[i])
        SequentialMonteCarlo.copy!(x_tpf[i][j], tpfs.X[1,j])
    end
end
##
trueX = hcat([p.x for p in ptrue]...)'
rtsX = hcat(data.ks.smooth_mean...)'
rtsσ = sqrt.(hcat(diag.(data.ks.smooth_Sigma)...))'

##
p2 = plot(trueX[:,1], ls=:dot, label="x_1", title="Posterior samples")
plot!(rtsX[:,1], ribbon=2*rtsσ[:,1], label="xhat_1")
plot!(map(p->p.x[1], x_tpf[1]), lw=1, lc=:black, la=0.1, label=false)
for i in 2:100:length(x)
    plot!(map(p->p.x[1], x_tpf[i]), lw=1, lc=:black, la=0.1,label=false)
end

plot(p2)


##

# X = view(bpfs.X, 1:M[1], :);
# ref = view(bpfs.ref, 1:T);
# A = view(bpfs.A, 1:M[1], :);
# wnorm = view(bpfs.wnorm, 1:M[1])

# println("After running bpf, before generating a reference trajectory")
# println("State trajectory matrix is ")
# println(X)

# finalInd = SequentialMonteCarlo.sample_one_index(wnorm); 
# Xref = SequentialMonteCarlo.generate_trajectory(A,X, ref, finalInd);  
# println("-------------------")
# println()
# println("After generating and storing a reference at index i=1")
# print("ref is ")
# println(ref)
# println("Reference state trajectory is ")
# println(X[1,:])
# println("State trajectory matrix is ")
# println(X)

# println("-------------------")
# println()
# println("Running CPF")
# Xref = bpf!(bpfs, model, data, θ; conditional =:yes, n_particles = M[1]);
# println("-------------------")
# println()
# println(" Done with CPF, new reference generated and stored")
# println("Reference trajectory is ")
# println(X[1,:])
# print("ref is ")
# println(ref)

# println("Done!")

#############################################################

# ##
# trueX = hcat([p.x for p in ptrue]...)'
# rtsX = hcat(data.ks.smooth_mean...)'
# rtsσ = sqrt.(hcat(diag.(data.ks.smooth_Sigma)...))'

# kfX = hcat(data.ks.filter_mean...)'
# kfσ = sqrt.(hcat(diag.(data.ks.filter_Sigma)...))'

# ## Plot Kalman filter estimate and true states
# p1 = plot(trueX[:,1], ls=:dot, label="x_1")
# plot!(kfX[:,1], ribbon=2*kfσ[:,1], label="xhat_1")
# p2 = plot(trueX[:,2], ls=:dot, label="x_2")
# plot!(kfX[:,2], ribbon=2*kfσ[:,2], label="xhat_2")
# p3 = plot(trueX[:,3], ls=:dot, label="x_3")
# plot!(kfX[:,3], ribbon=2*kfσ[:,3], label="xhat_3")
# plot(p1, p2, p3, title="Kalman filter estimates")

# ## Plot RTS smoother estimate and true states
# p1 = plot(trueX[:,1], ls=:dot, label="x_1")
# plot!(rtsX[:,1], ribbon=2*rtsσ[:,1], label="xhat_1")
# p2 = plot(trueX[:,2], ls=:dot, label="x_2")
# plot!(rtsX[:,2], ribbon=2*rtsσ[:,2], label="xhat_2")
# p3 = plot(trueX[:,3], ls=:dot, label="x_3")
# plot!(rtsX[:,3], ribbon=2*rtsσ[:,3], label="xhat_3")
# plot(p1, p2, p3, title="RTS smother")

# ##
# Wnorm = exp.(bpfs.W .- maximum(bpfs.W, dims=1))
# for i in 1:size(Wnorm, 2)
#     @views normalize!(Wnorm[:,i], 1)
# end

# Xbpf = map(SequentialMonteCarlo.toSVector, bpfs.X)
# bpfX = Array(hcat(sum(Wnorm .* Xbpf, dims=1)...)')

# bpfσ = similar(bpfX)
# for i in 1:size(bpfσ, 1)
#     @views bpfσ[i,:] .= sqrt.(sum(Wnorm[:,i] .* map(x->(x.-bpfX[i,:]).^2, Xbpf[:,i])))
# end

# Wnorm = exp.(tpfs.W .- maximum(tpfs.W, dims=1))
# for i in 1:size(Wnorm, 2)
#     @views normalize!(Wnorm[:,i], 1)
# end

# Xtpf = map(SequentialMonteCarlo.toSVector, tpfs.X)
# tpfX = Array(hcat(sum(Wnorm .* Xtpf, dims=1)...)')

# tpfσ = similar(tpfX)
# for i in 1:size(tpfσ, 1)
#     @views tpfσ[i,:] .= sqrt.(sum(Wnorm[:,i] .* map(x->(x.-tpfX[i,:]).^2, Xtpf[:,i])))
# end

# ## Bootstrap PF
# p1 = plot(bpfX[:,1], label="bpf", ribbon=2*bpfσ[:,1], fillalpha=0.5)
# plot!(kfX[:,1], label="kf", ribbon=2*kfσ[:,1], fillalpha=0.5)
# p2 = plot(bpfX[:,2], label="bpf", ribbon=2*bpfσ[:,2], fillalpha=0.5)
# plot!(kfX[:,2], label="kf", ribbon=2*kfσ[:,2], fillalpha=0.5)
# p3 = plot(bpfX[:,3], label="bpf", ribbon=2*bpfσ[:,3], fillalpha=0.5)
# plot!(kfX[:,3], label="kf", ribbon=2*kfσ[:,3], fillalpha=0.5)
# plot(p1, p2, p3)

# ## Twisted PF
# p4 = plot(tpfX[:,1], label="tpf", ribbon=2*tpfσ[:,1], fillalpha=0.5)
# plot!(rtsX[:,1], label="rts", ribbon=2*rtsσ[:,1], fillalpha=0.5)
# p5 = plot(tpfX[:,2], label="tpf", ribbon=2*tpfσ[:,2], fillalpha=0.5)
# plot!(rtsX[:,2], label="rts", ribbon=2*rtsσ[:,2], fillalpha=0.5)
# p6 = plot(tpfX[:,3], label="tpf", ribbon=2*tpfσ[:,3], fillalpha=0.5)
# plot!(rtsX[:,3], label="rts", ribbon=2*rtsσ[:,3], fillalpha=0.5)
# plot(p4, p5, p6)

# ##

# # Estimate the likelihood N_mc times for each number of particles in M  using the particle filter
# ll = zeros(length(M), N_mc)
# for i in eachindex(M)
#     println(M[i], " particles")
#     for j in ProgressBar(1:size(ll,2))
#         ll[i, j] = bpf!(bpfs, model, data, θ; n_particles = M[i])
#     end
# end

# df = DataFrame(ll')
# rename!(df, Symbol.(M))
# df = stack(df)
# insertcols!(df, :method=>["Bootstrap" for i = 1:size(df, 1)])

# ll_tpf = zeros(length(M_tpf), N_mc)
# for i in eachindex(M_tpf)
#     println("Twisted particle filter with ", M_tpf[i], " particles")
#     for j in ProgressBar(1:size(ll_tpf, 2))
#         ll_tpf[i, j] = tpf!(tpfs, model, data, θ; n_particles = M_tpf[i])
#     end
# end

# df_tpf = DataFrame(ll_tpf')
# rename!(df_tpf, Symbol.(M_tpf))
# df_tpf = stack(df_tpf)
# insertcols!(df_tpf, :method=>["Twisted" for i = 1:size(df_tpf, 1)])

# append!(df, df_tpf)
# ##
# # @df df violin(string.(:method), :value, title="Likelihood estimates bootstrap PF", legend=false)
# p1 = @df filter(row -> row[:method] == "Bootstrap", df) violin(
#     string.(:variable),
#     :value,
#     side=:left,
#     label="Bootstrap",
#     legend=true,
#     # xticks=(1:length(M), string.(M_tpf))
#     )
# @df filter(row -> row[:method] == "Twisted", df) violin!(string.(:variable), :value, side=:right, label="Twisted")
# # boxplot!(ll_tpf', legend=false, xticks=(1:length(M), string.(M_tpf)))
# hline!([data.ks.log_likelihood[end]], color="black", width=2, ls=:dash, label="Z")

# p2 = boxplot(ll_tpf', title="Likelihood estimates twisted PF", legend=false, xticks=(1:length(M), string.(M_tpf)), xlabel="Number of particles")
# hline!([data.ks.log_likelihood[end]], color="black", width=2, ls=:dash, label="Z")

# plot(p1, p2)
