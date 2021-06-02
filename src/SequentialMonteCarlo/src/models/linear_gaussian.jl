"""
A linear Gaussian state-space model parameterized by the state dimension
```math
xₜ₊₁ = Axₜ + Buₜ + vₜ \\
  yₜ = Cxₜ + Duₜ + eₜ \\
where vₜ ~ N(0, Q), eₜ ~ N(0, R) and x₁ ~ N(μ0, Σ0)
```
"""
struct LinearGaussian{nx,ny,nu,T} <: AbstractSSM{FloatParticle{nx, T}} end
LinearGaussian{nx,ny,nu}() where {nx,ny,nu} = LinearGaussian{nx,ny,nu,Float64}()

Base.eltype(::LinearGaussian{<:Any,<:Any,<:Any,T}) where T = T
output_dimension(::LinearGaussian{<:Any,ny}) where ny = ny

@with_kw struct LinearGaussianPar{AT<:AbstractMatrix, BT<:Union{AbstractMatrix,Val{false}},
    CT<:AbstractMatrix, DT<:Union{AbstractMatrix,Val{false}}, QT<:PDMat, RT<:PDMat, IT, ΣT<:PDMat}
    A::AT; @assert size(A,1) === size(A,2)
    B::BT = Val(false); @assert B === Val(false) || size(B,1) === size(A,1)
    C::CT; @assert size(C,2) === size(A,1)
    D::DT = Val(false); @assert D === Val(false) || (size(D,1)  === size(C,1) && (B === Val(false) || size(D,2) === size(B,2)))
    Q::QT; @assert size(Q) == size(A)
    R::RT; @assert size(R,1) == size(R,2) == size(C,1)
    μ₀::IT; @assert size(μ₀,1) == size(A,1)
    Σ₀::ΣT; @assert size(Σ₀,1) == size(Σ₀,2) && size(Σ₀,1) == size(A, 1)
end

function random_parameters(::LinearGaussian{nx,ny,nu,T}) where {nx,ny,nu,T}
    A = randn(SMatrix{nx,nx,T}) |> qr |> x->x.Q*diagm(-1 .+ 2.0.*rand(SVector{nx,T}))*x.Q';
    B = randn(SMatrix{nx,nu,T});
    C = randn(SMatrix{ny,nx,T})
    D = Val(false)
    Q = randn(SMatrix{nx,nx,T}) |> x->x*x'+0.1*one(SMatrix{nx,nx,T}) |> PDMat
    R = randn(SMatrix{ny,ny,T}) |> x->x*x'+0.1*one(SMatrix{ny,ny,T}) |> PDMat
    μ₀ = randn(SVector{nx,T})
    Σ₀ = randn(SMatrix{nx,nx,T}) |> x->x*x'+0.1*one(SMatrix{nx,nx,T}) |> PDMat
    LinearGaussianPar(A,B,C,D,Q,R,μ₀,Σ₀)
end

initial_density(::LinearGaussian, data, θ) = MvNormal(θ.μ₀, θ.Σ₀)
transition_noise(xₜ, ::LinearGaussian, t, data, θ) = MvNormal(θ.Q)
observation_noise(xₜ, ::LinearGaussian, t, data, θ) = MvNormal(θ.R)

@inline function transition_function(x, v, ::LinearGaussian, t, data, θ)
    @unpack A, B = θ
    return A*x + B*data.u[t] + v
end

@inline function transition_function(x, v, ::LinearGaussian, t, data, θ::LinearGaussianPar{<:Any,Val{false}})
    @unpack A = θ
    return A*x + v
end

@inline function observation_function(x, e, ::LinearGaussian, t, data, θ)
    @unpack C, D = θ
    return C*x + D*data.u[t] + e
end

@inline function observation_function(x, e, ::LinearGaussian, t, data, θ::LinearGaussianPar{<:Any,<:Any,<:Any,Val{false}})
    @unpack C = θ
    return C*x + e
end

############################
###       OPTIONAL       ###
############################


### Jacobians ###
transition_state_jacobian(x, ::LinearGaussian, t, data, θ) = θ.A
transition_noise_jacobian(x, ::LinearGaussian, t, data, θ) = one(θ.Q.mat)

observation_state_jacobian(x, ::LinearGaussian, t, data, θ) = θ.C
observation_noise_jacobian(x, ::LinearGaussian, t, data, θ) = one(θ.R.mat)
