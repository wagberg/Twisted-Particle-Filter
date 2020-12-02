const CoupledTankParticle = FloatParticle{2}

"""
Coupled tank model

xₜ₊₁ = 

xₜ | xₜ₋₁ ~ N(xₜ ; ϕxₜ₋₁, σ²)
yₜ | xₜ ~N(yₜ ; 0, β²exp(xₜ))
x₁ ~ N(x₁ ; μ₁, σ₁²)

Parameters:
* `μ₁` - Initial state mean
* `logσ₁` - Initial state std log(σ₁)
* `logϕ` - log(ϕ)
* `logσ` - log(σ)
* `logβ` - log(β)
"""
struct CoupledTank <: SSM{CoupledTankParticle}
end

@with_kw struct TankParameter
    μ0::SVector{2} = [6.225503938785832, 4.9728]
    Σ0::Symmetric = Symmetric(Matrix{Float64}(I,2,2))
    logk₁::Float64 = log(0.054115135099972)
    logk₂::Float64 = log(0.069031758496706)
    logk₃::Float64 = log(0.044688311520571)
    logk₄::Float64 = log(0.224558186566615)
    logk₅::Float64 = log(0.002262430348693)
    logk₆::Float64 = log(0.006535014001622)
    logσw²::Float64 = log(0.002513535972694)
    logσe²::Float64 = log(0.013531902670792)
    Ts::Float64 = 4.0
end

# @inline function UnPack.unpack(x::TankParameter, ::Val{f}) where {f}
#     exp(getproperty(x, f))
# end


function SequentialMonteCarlo.transition_function(x, ::CoupledTank, t, data, θ)
    @unpack logk₁, logk₂, logk₃, logk₄, logk₅, logk₆, Ts  = θ
    xp = clamp.(x, 0.0, 10.0)
    @inbounds xp[1] += Ts*(-exp(logk₁)*sqrt(xp[1]) - exp(logk₅)*xp[1] + exp(logk₃)*data.u[t][1])
    @inbounds xp[2] += Ts*(exp(logk₁)*sqrt(xp[1]) + exp(logk₅)*xp[1] - exp(logk₂)*sqrt(xp[2]) - exp(logk₆)*xp[2] + exp(logk₄)*max.(x[1]-10.0, 0.0))
    return xp
end

function SequentialMonteCarlo.observation_function(x, ::CoupledTank, t, data, θ)
    clamp.(x[2:2], 0.0, 10.0)
end

function SequentialMonteCarlo.transition_covariance(x, ::CoupledTank, t, data, θ)
    @unpack logσw² = θ
    exp(logσw²)*I(2)
end

function SequentialMonteCarlo.observation_covariance(x, ::CoupledTank, t, data, θ)
    @unpack logσe² = θ
    exp(logσe²)* I(1)
end